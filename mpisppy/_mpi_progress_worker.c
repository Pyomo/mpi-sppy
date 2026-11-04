/* GIL-free MPI progress worker. Build against the MPI used by mpi4py. */
#define _POSIX_C_SOURCE 200809L
#include <Python.h>
#include <errno.h>
#include <math.h>
#include <mpi4py/mpi4py.h>
#include "mpi_progress_build.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STOP_TIMEOUT_SECONDS 30

/* macOS has no pthread_condattr_setclock. Its default condition clock is
   CLOCK_REALTIME, so timed waits must use that clock too. */
#ifdef __linux__
#define PROGRESS_COND_CLOCK CLOCK_MONOTONIC
#else
#define PROGRESS_COND_CLOCK CLOCK_REALTIME
#endif

static struct {
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    int configured;
    int running;
    int active;
    int polling;
    int stopped;
    int mpi_error;
    int wait_error;
    MPI_Comm comm;
    PyObject *comm_owner;
    double max_interval;
} worker = {.mutex = PTHREAD_MUTEX_INITIALIZER};

static double seconds_on(clockid_t clock_id)
{
    struct timespec ts;
    clock_gettime(clock_id, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static struct timespec deadline_after(double seconds)
{
    struct timespec deadline;
    long nanoseconds;
    clock_gettime(PROGRESS_COND_CLOCK, &deadline);
    deadline.tv_sec += (time_t)seconds;
    nanoseconds = (long)((seconds - (time_t)seconds) * 1e9);
    deadline.tv_nsec += nanoseconds;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    return deadline;
}

static void abort_worker(const char *operation, int error)
{
    fprintf(stderr, "mpi-sppy MPI progress: %s failed with error %d; aborting MPI job\n",
            operation, error);
    fflush(stderr);
    MPI_Abort(worker.comm, EXIT_FAILURE);
    _Exit(EXIT_FAILURE); /* MPI_Abort must not return. */
}

static void exit_stalled_worker(const char *operation)
{
    fprintf(stderr, "mpi-sppy MPI progress: %s exceeded %d seconds; exiting rank\n",
            operation, STOP_TIMEOUT_SECONDS);
    fflush(stderr);
    _Exit(EXIT_FAILURE); /* Never finalize MPI with a live progress worker. */
}

static PyObject *configure(PyObject *self, PyObject *args)
{
    PyObject *comm_obj;
    MPI_Comm *comm_ptr;
    double interval;
    int error;
#ifdef __linux__
    pthread_condattr_t attributes;
#endif

    if (!PyArg_ParseTuple(args, "Od", &comm_obj, &interval))
        return NULL;
    if (!isfinite(interval) || interval <= 0) {
        PyErr_SetString(PyExc_ValueError, "progress interval must be positive and finite");
        return NULL;
    }
    comm_ptr = PyMPIComm_Get(comm_obj);
    if (comm_ptr == NULL)
        return NULL;
    pthread_mutex_lock(&worker.mutex);
    if (worker.configured) {
        pthread_mutex_unlock(&worker.mutex);
        PyErr_SetString(PyExc_RuntimeError, "MPI progress worker is already configured");
        return NULL;
    }
#ifdef __linux__
    error = pthread_condattr_init(&attributes);
    if (error == 0) {
        error = pthread_condattr_setclock(&attributes, PROGRESS_COND_CLOCK);
        if (error == 0)
            error = pthread_cond_init(&worker.changed, &attributes);
        pthread_condattr_destroy(&attributes);
    }
#else
    error = pthread_cond_init(&worker.changed, NULL);
#endif
    if (error != 0) {
        pthread_mutex_unlock(&worker.mutex);
        PyErr_Format(PyExc_OSError, "could not initialize progress condition: %d", error);
        return NULL;
    }
    Py_INCREF(comm_obj);
    worker.comm_owner = comm_obj;
    worker.comm = *comm_ptr;
    worker.max_interval = interval;
    worker.configured = 1;
    pthread_mutex_unlock(&worker.mutex);
    Py_RETURN_NONE;
}

static PyObject *run(PyObject *self, PyObject *unused)
{
    pthread_mutex_lock(&worker.mutex);
    if (worker.configured && worker.stopped) {
        pthread_mutex_unlock(&worker.mutex);
        Py_RETURN_NONE; /* stop() can run before this thread starts. */
    }
    if (!worker.configured || worker.running) {
        pthread_mutex_unlock(&worker.mutex);
        PyErr_SetString(PyExc_RuntimeError, "MPI progress worker is unavailable");
        return NULL;
    }
    worker.running = 1;
    pthread_mutex_unlock(&worker.mutex);

    /* No Python operation or GIL transition occurs inside the polling loop. */
    PyThreadState *thread_state = PyEval_SaveThread();
    for (;;) {
        double active_start;
        int wait_error;
        pthread_mutex_lock(&worker.mutex);
        while (!worker.active && !worker.stopped) {
            wait_error = pthread_cond_wait(&worker.changed, &worker.mutex);
            if (wait_error != 0) {
                worker.wait_error = wait_error;
                worker.stopped = 1;
                pthread_cond_broadcast(&worker.changed);
                pthread_mutex_unlock(&worker.mutex);
                abort_worker("pthread_cond_wait", wait_error);
            }
        }
        if (worker.stopped) {
            pthread_mutex_unlock(&worker.mutex);
            break;
        }
        active_start = seconds_on(CLOCK_MONOTONIC);
        worker.polling = 1;
        pthread_mutex_unlock(&worker.mutex);

        for (;;) {
            double probe_end, elapsed, interval;
            struct timespec deadline;
            int flag, mpi_error, wait_error = 0;
            pthread_mutex_lock(&worker.mutex);
            if (!worker.active || worker.stopped) {
                worker.polling = 0;
                pthread_cond_broadcast(&worker.changed);
                pthread_mutex_unlock(&worker.mutex);
                break;
            }
            pthread_mutex_unlock(&worker.mutex);

            mpi_error = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, worker.comm,
                                   &flag, MPI_STATUS_IGNORE);
            probe_end = seconds_on(CLOCK_MONOTONIC);
            if (mpi_error != MPI_SUCCESS) {
                pthread_mutex_lock(&worker.mutex);
                worker.mpi_error = mpi_error;
                worker.stopped = 1;
                pthread_cond_broadcast(&worker.changed);
                pthread_mutex_unlock(&worker.mutex);
                abort_worker("MPI_Iprobe", mpi_error);
            }
            elapsed = probe_end - active_start;
            interval = worker.max_interval;
            if (elapsed < 0.010 && interval > 50e-6)
                interval = 50e-6;
            else if (elapsed < 0.100 && interval > 250e-6)
                interval = 250e-6;
            deadline = deadline_after(interval);
            pthread_mutex_lock(&worker.mutex);
            if (worker.active && !worker.stopped) {
                wait_error = pthread_cond_timedwait(
                    &worker.changed, &worker.mutex, &deadline);
                if (wait_error != 0 && wait_error != ETIMEDOUT) {
                    worker.wait_error = wait_error;
                    worker.stopped = 1;
                    pthread_cond_broadcast(&worker.changed);
                }
            }
            pthread_mutex_unlock(&worker.mutex);
            if (wait_error != 0 && wait_error != ETIMEDOUT)
                abort_worker("pthread_cond_timedwait", wait_error);
        }
    }
    pthread_mutex_lock(&worker.mutex);
    worker.running = 0;
    pthread_cond_broadcast(&worker.changed);
    pthread_mutex_unlock(&worker.mutex);
    PyEval_RestoreThread(thread_state);
    Py_RETURN_NONE;
}

static PyObject *set_active(PyObject *self, PyObject *args)
{
    int active;
    int unavailable, mpi_error, wait_error, condition_error = 0, stalled;
    PyThreadState *thread_state;
    struct timespec deadline;
    if (!PyArg_ParseTuple(args, "p", &active))
        return NULL;
    /* A deactivation waits until no MPI call remains in the worker. */
    thread_state = PyEval_SaveThread();
    pthread_mutex_lock(&worker.mutex);
    if (worker.configured && !worker.stopped) {
        worker.active = active;
        pthread_cond_signal(&worker.changed);
        if (!active && worker.polling) {
            deadline = deadline_after(STOP_TIMEOUT_SECONDS);
            while (worker.polling && condition_error == 0)
                condition_error = pthread_cond_timedwait(
                    &worker.changed, &worker.mutex, &deadline);
        }
    }
    stalled = condition_error == ETIMEDOUT && worker.polling;
    unavailable = !worker.configured || worker.stopped;
    mpi_error = worker.mpi_error;
    wait_error = worker.wait_error;
    pthread_mutex_unlock(&worker.mutex);
    if (stalled)
        exit_stalled_worker("progress deactivation");
    if (condition_error != 0 && condition_error != ETIMEDOUT)
        abort_worker("pthread_cond_timedwait during deactivation", condition_error);
    PyEval_RestoreThread(thread_state);
    if (unavailable) {
        PyErr_Format(PyExc_RuntimeError,
                     "MPI progress worker stopped (MPI error %d, wait error %d)",
                     mpi_error, wait_error);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *stop(PyObject *self, PyObject *unused)
{
    pthread_mutex_lock(&worker.mutex);
    if (worker.configured) {
        worker.stopped = 1;
        worker.active = 0;
        pthread_cond_broadcast(&worker.changed);
    }
    pthread_mutex_unlock(&worker.mutex);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"configure", configure, METH_VARARGS, "Configure the native progress worker."},
    {"run", run, METH_NOARGS, "Run MPI progress without per-probe GIL acquisition."},
    {"set_active", set_active, METH_VARARGS, "Start or pause progress polling."},
    {"stop", stop, METH_NOARGS, "Stop and wake the progress worker."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_mpi_progress_worker", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__mpi_progress_worker(void)
{
    PyObject *result;
    if (import_mpi4py() < 0)
        return NULL;
    result = PyModule_Create(&module);
    if (result == NULL)
        return NULL;
    if (PyModule_AddIntConstant(result, "STOP_TIMEOUT_SECONDS", STOP_TIMEOUT_SECONDS) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    if (PyModule_AddStringConstant(result, "BUILD_MPI_LIBRARY_VERSION",
                                   mpisppy_build_mpi_library_version) < 0 ||
        PyModule_AddIntConstant(result, "BUILD_MPI_COMM_SIZE",
                                MPISPPY_BUILD_MPI_COMM_SIZE) < 0 ||
        PyModule_AddIntConstant(result, "BUILD_MPI_ANY_SOURCE",
                                MPISPPY_BUILD_MPI_ANY_SOURCE) < 0 ||
        PyModule_AddIntConstant(result, "BUILD_MPI_ANY_TAG",
                                MPISPPY_BUILD_MPI_ANY_TAG) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
