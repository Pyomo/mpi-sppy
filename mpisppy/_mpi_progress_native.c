/* Optional diagnostic helper. Build with the same MPI implementation as mpi4py. */
#define _POSIX_C_SOURCE 200809L
#include <Python.h>
#include <errno.h>
#include <math.h>
#include <mpi4py/mpi4py.h>
#include "mpi_progress_build.h"
#include <sys/select.h>
#include <time.h>

static double monotonic_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* The timestamps surrounding MPI_Iprobe are taken without the Python GIL. */
static PyObject *timed_iprobe(PyObject *self, PyObject *args)
{
    PyObject *comm_obj;
    MPI_Comm *comm_ptr;
    int source, tag, flag = 0, error;
    double mpi_start, mpi_end, gil_end;
    PyThreadState *state;

    if (!PyArg_ParseTuple(args, "Oii", &comm_obj, &source, &tag))
        return NULL;
    comm_ptr = PyMPIComm_Get(comm_obj);
    if (comm_ptr == NULL)
        return NULL;

    state = PyEval_SaveThread();
    mpi_start = monotonic_seconds();
    error = MPI_Iprobe(source, tag, *comm_ptr, &flag, MPI_STATUS_IGNORE);
    mpi_end = monotonic_seconds();
    PyEval_RestoreThread(state);
    gil_end = monotonic_seconds();
    if (error != MPI_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "MPI_Iprobe failed with MPI error %d", error);
        return NULL;
    }
    return Py_BuildValue("idd", flag, mpi_end - mpi_start, gil_end - mpi_end);
}

/* Wait on a stop pipe, recording the return from pselect before taking the GIL. */
static PyObject *timed_wait(PyObject *self, PyObject *args)
{
    int stop_fd, stopped = 0, error = 0, ready;
    double seconds, start, deadline, now, wait_end, gil_end;
    struct timespec timeout;
    fd_set readfds;
    PyThreadState *state;

    if (!PyArg_ParseTuple(args, "id", &stop_fd, &seconds))
        return NULL;
    if (stop_fd < 0 || stop_fd >= FD_SETSIZE || !isfinite(seconds) || seconds < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid stop fd or wait interval");
        return NULL;
    }

    state = PyEval_SaveThread();
    start = monotonic_seconds();
    deadline = start + seconds;
    for (;;) {
        now = monotonic_seconds();
        if (now >= deadline)
            break;
        timeout.tv_sec = (time_t)(deadline - now);
        timeout.tv_nsec = (long)((deadline - now - (double)timeout.tv_sec) * 1e9);
        FD_ZERO(&readfds);
        FD_SET(stop_fd, &readfds);
        ready = pselect(stop_fd + 1, &readfds, NULL, NULL, &timeout, NULL);
        if (ready > 0) {
            stopped = 1;
            break;
        }
        if (ready == 0)
            continue; /* Account for timer rounding before the deadline. */
        if (errno != EINTR) {
            error = errno;
            break;
        }
    }
    wait_end = monotonic_seconds();
    PyEval_RestoreThread(state);
    gil_end = monotonic_seconds();
    if (error) {
        errno = error;
        return PyErr_SetFromErrno(PyExc_OSError);
    }
    return Py_BuildValue("idd", stopped, wait_end - start, gil_end - wait_end);
}

static PyMethodDef methods[] = {
    {"timed_iprobe", timed_iprobe, METH_VARARGS, "Time MPI_Iprobe and GIL reacquisition."},
    {"timed_wait", timed_wait, METH_VARARGS, "Time a stop-aware wait and GIL reacquisition."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_mpi_progress_native", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__mpi_progress_native(void)
{
    PyObject *result;
    if (import_mpi4py() < 0)
        return NULL;
    result = PyModule_Create(&module);
    if (result == NULL)
        return NULL;
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
