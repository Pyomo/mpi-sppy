/* GIL-free MPI progress worker. Build against the MPI used by mpi4py. */
#define _POSIX_C_SOURCE 200809L
#include <Python.h>
#include <errno.h>
#include <math.h>
#include <mpi4py/mpi4py.h>
#include <pthread.h>
#include <time.h>

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
    int diagnostics;
    MPI_Comm comm;
    PyObject *comm_owner;
    double max_interval;
    unsigned long long probes;
    unsigned long long wakeups;
    double active_seconds;
    double mpi_seconds;
    double max_probe_seconds;
    double wait_late_seconds;
    double max_wait_late_seconds;
    double worker_cpu_seconds;
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
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += (time_t)seconds;
    nanoseconds = (long)((seconds - (time_t)seconds) * 1e9);
    deadline.tv_nsec += nanoseconds;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    return deadline;
}

static double deadline_seconds(struct timespec deadline)
{
    return (double)deadline.tv_sec + (double)deadline.tv_nsec * 1e-9;
}

static PyObject *configure(PyObject *self, PyObject *args)
{
    PyObject *comm_obj;
    MPI_Comm *comm_ptr;
    double interval;
    int diagnostics;
    pthread_condattr_t attributes;
    int error;

    if (!PyArg_ParseTuple(args, "Odp", &comm_obj, &interval, &diagnostics))
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
    error = pthread_condattr_init(&attributes);
    if (error == 0) {
        error = pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC);
        if (error == 0)
            error = pthread_cond_init(&worker.changed, &attributes);
        pthread_condattr_destroy(&attributes);
    }
    if (error != 0) {
        pthread_mutex_unlock(&worker.mutex);
        PyErr_Format(PyExc_OSError, "could not initialize progress condition: %d", error);
        return NULL;
    }
    Py_INCREF(comm_obj);
    worker.comm_owner = comm_obj;
    worker.comm = *comm_ptr;
    worker.max_interval = interval;
    worker.diagnostics = diagnostics;
    worker.configured = 1;
    pthread_mutex_unlock(&worker.mutex);
    Py_RETURN_NONE;
}

static PyObject *run(PyObject *self, PyObject *unused)
{
    double cpu_start;
    int mpi_error = MPI_SUCCESS;
    int wait_error = 0;
    pthread_mutex_lock(&worker.mutex);
    if (!worker.configured || worker.running || worker.stopped) {
        pthread_mutex_unlock(&worker.mutex);
        PyErr_SetString(PyExc_RuntimeError, "MPI progress worker is unavailable");
        return NULL;
    }
    worker.running = 1;
    pthread_mutex_unlock(&worker.mutex);

    /* No Python operation or GIL transition occurs inside the polling loop. */
    PyThreadState *thread_state = PyEval_SaveThread();
    cpu_start = worker.diagnostics ? seconds_on(CLOCK_THREAD_CPUTIME_ID) : 0;
    for (;;) {
        double active_start;
        pthread_mutex_lock(&worker.mutex);
        while (!worker.active && !worker.stopped)
            pthread_cond_wait(&worker.changed, &worker.mutex);
        if (worker.stopped) {
            pthread_mutex_unlock(&worker.mutex);
            break;
        }
        active_start = seconds_on(CLOCK_MONOTONIC);
        worker.polling = 1;
        if (worker.diagnostics)
            worker.wakeups++;
        pthread_mutex_unlock(&worker.mutex);

        for (;;) {
            double probe_start = 0.0, probe_end, elapsed, interval, lateness;
            struct timespec deadline;
            int flag;
            pthread_mutex_lock(&worker.mutex);
            if (!worker.active || worker.stopped) {
                if (worker.diagnostics)
                    worker.active_seconds += seconds_on(CLOCK_MONOTONIC) - active_start;
                worker.polling = 0;
                pthread_cond_broadcast(&worker.changed);
                pthread_mutex_unlock(&worker.mutex);
                break;
            }
            pthread_mutex_unlock(&worker.mutex);

            if (worker.diagnostics)
                probe_start = seconds_on(CLOCK_MONOTONIC);
            mpi_error = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, worker.comm,
                                   &flag, MPI_STATUS_IGNORE);
            probe_end = seconds_on(CLOCK_MONOTONIC);
            if (mpi_error != MPI_SUCCESS) {
                pthread_mutex_lock(&worker.mutex);
                worker.mpi_error = mpi_error;
                worker.stopped = 1;
                pthread_cond_broadcast(&worker.changed);
                pthread_mutex_unlock(&worker.mutex);
                continue;
            }
            if (worker.diagnostics) {
                worker.probes++;
                worker.mpi_seconds += probe_end - probe_start;
                if (probe_end - probe_start > worker.max_probe_seconds)
                    worker.max_probe_seconds = probe_end - probe_start;
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
            if (worker.diagnostics && worker.active && !worker.stopped) {
                lateness = seconds_on(CLOCK_MONOTONIC) - deadline_seconds(deadline);
                if (lateness > 0) {
                    worker.wait_late_seconds += lateness;
                    if (lateness > worker.max_wait_late_seconds)
                        worker.max_wait_late_seconds = lateness;
                }
            }
            pthread_mutex_unlock(&worker.mutex);
        }
    }
    pthread_mutex_lock(&worker.mutex);
    if (worker.diagnostics)
        worker.worker_cpu_seconds = seconds_on(CLOCK_THREAD_CPUTIME_ID) - cpu_start;
    worker.running = 0;
    pthread_cond_broadcast(&worker.changed);
    pthread_mutex_unlock(&worker.mutex);
    PyEval_RestoreThread(thread_state);
    if (mpi_error != MPI_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "MPI_Iprobe failed with MPI error %d", mpi_error);
        return NULL;
    }
    if (wait_error != 0 && wait_error != ETIMEDOUT) {
        PyErr_Format(PyExc_OSError, "MPI progress wait failed with error %d", wait_error);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *set_active(PyObject *self, PyObject *args)
{
    int active;
    int unavailable, mpi_error, wait_error;
    PyThreadState *thread_state;
    if (!PyArg_ParseTuple(args, "p", &active))
        return NULL;
    /* A deactivation waits until no MPI call remains in the worker. */
    thread_state = PyEval_SaveThread();
    pthread_mutex_lock(&worker.mutex);
    unavailable = !worker.configured || worker.stopped;
    mpi_error = worker.mpi_error;
    wait_error = worker.wait_error;
    if (!unavailable) {
        worker.active = active;
        pthread_cond_signal(&worker.changed);
        while (!active && worker.polling)
            pthread_cond_wait(&worker.changed, &worker.mutex);
    }
    pthread_mutex_unlock(&worker.mutex);
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

static PyObject *snapshot(PyObject *self, PyObject *unused)
{
    PyObject *result;
    pthread_mutex_lock(&worker.mutex);
    if (worker.running) {
        pthread_mutex_unlock(&worker.mutex);
        PyErr_SetString(PyExc_RuntimeError, "join MPI progress worker before reading counters");
        return NULL;
    }
    result = Py_BuildValue("{s:K,s:K,s:d,s:d,s:d,s:d,s:d,s:d,s:i,s:i}",
        "probes", worker.probes,
        "wakeups", worker.wakeups,
        "active_seconds", worker.active_seconds,
        "mpi_call_seconds", worker.mpi_seconds,
        "max_probe_seconds", worker.max_probe_seconds,
        "wait_return_late_seconds", worker.wait_late_seconds,
        "max_wait_return_late_seconds", worker.max_wait_late_seconds,
        "worker_thread_cpu_seconds", worker.worker_cpu_seconds,
        "mpi_error", worker.mpi_error,
        "wait_error", worker.wait_error);
    pthread_mutex_unlock(&worker.mutex);
    return result;
}

static PyMethodDef methods[] = {
    {"configure", configure, METH_VARARGS, "Configure the native progress worker."},
    {"run", run, METH_NOARGS, "Run MPI progress without per-probe GIL acquisition."},
    {"set_active", set_active, METH_VARARGS, "Start or pause progress polling."},
    {"stop", stop, METH_NOARGS, "Stop and wake the progress worker."},
    {"snapshot", snapshot, METH_NOARGS, "Read counters after the worker has stopped."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_mpi_progress_worker", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__mpi_progress_worker(void)
{
    if (import_mpi4py() < 0)
        return NULL;
    return PyModule_Create(&module);
}
