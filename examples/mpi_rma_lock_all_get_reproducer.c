/*
 * Minimal cross-node MPI_Get reproducer using one persistent lock-all epoch.
 * Rank 0 exposes one static double; rank 1 repeatedly reads it.
 *
 * Build from this directory with: make
 * Run one rank per node with:
 *   srun -u -N 2 -n 2 --ntasks-per-node=1 \
 *       ./mpi_rma_lock_all_get_reproducer 100
 */

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

enum { DEFAULT_ITERATIONS = 100 };

static int parse_iterations(const char *text, int rank)
{
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
            value <= 0 || value > INT_MAX) {
        if (rank == 0) {
            fprintf(stderr, "ITERATIONS must be a positive integer; got %s\n",
                    text);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return (int)value;
}

int main(int argc, char **argv)
{
    int rank;
    int size;
    int iterations = DEFAULT_ITERATIONS;
    int model_found = 0;
    int *window_model = NULL;
    MPI_Aint window_nbytes;
    MPI_Win window = MPI_WIN_NULL;
    double *window_base = NULL;
    double received = -1.0;
    char processor[MPI_MAX_PROCESSOR_NAME];
    int processor_length;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 2) {
        if (rank == 0) {
            fprintf(stderr, "usage: %s [ITERATIONS]\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (argc == 2) {
        iterations = parse_iterations(argv[1], rank);
    }
    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "world size must be two; got %d\n", size);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    window_nbytes = rank == 0 ? (MPI_Aint)sizeof(*window_base) : 0;
    MPI_Win_allocate(window_nbytes, 1, MPI_INFO_NULL, MPI_COMM_WORLD,
                     &window_base, &window);

    if (rank == 0) {
        *window_base = 42.0;
    }

    MPI_Win_lock_all(0, window);
    MPI_Win_sync(window);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Get_processor_name(processor, &processor_length);
    printf("rank=%d host=%.*s\n", rank, processor_length, processor);
    fflush(stdout);

    MPI_Win_get_attr(window, MPI_WIN_MODEL, &window_model, &model_found);
    if (rank == 0) {
        printf("world=2 iterations=%d window_model=%s\n",
               iterations,
               model_found && *window_model == MPI_WIN_UNIFIED
                   ? "unified" : "separate-or-unknown");
        fflush(stdout);
    }

    if (rank == 1) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            MPI_Get(&received, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, window);
            MPI_Win_flush(0, window);
            if (received != 42.0) {
                fprintf(stderr,
                        "rank=1 iteration=%d expected=42 actual=%.17g\n",
                        iteration, received);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            printf("completed %d iterations\n", iteration + 1);
            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_unlock_all(window);
    MPI_Win_free(&window);

    if (rank == 0) {
        puts("PASS");
    }
    MPI_Finalize();
    return 0;
}
