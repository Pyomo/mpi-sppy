/*
 * Minimal C translation of mpi4py_xhat_collective_reproducer.py --mode
 * get-only. Rank 1 repeatedly gets one double from rank 0.
 *
 * Compile:
 *   mpicc -O2 -Wall -Wextra -o mpi_rma_get_reproducer \
 *       examples/mpi_rma_get_reproducer.c
 *
 * Run one rank on each of two nodes:
 *   srun -u -N 2 -n 2 --ntasks-per-node=1 ./mpi_rma_get_reproducer 100
 *
 * ITERATIONS is optional and defaults to 100.
 */

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

enum { DEFAULT_ITERATIONS = 100 };

static int parse_positive_int(const char *text, const char *name, int world_rank)
{
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > INT_MAX) {
        if (world_rank == 0) {
            fprintf(stderr, "%s must be a positive integer; got %s\n", name, text);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return (int)value;
}

int main(int argc, char **argv)
{
    int world_rank;
    int world_size;
    int iterations = DEFAULT_ITERATIONS;
    MPI_Win window = MPI_WIN_NULL;
    double *window_base = NULL;
    double initial;
    double received;
    int *window_model = NULL;
    int model_found = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc > 2) {
        if (world_rank == 0) {
            fprintf(stderr, "usage: %s [ITERATIONS]\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (argc >= 2) {
        iterations = parse_positive_int(argv[1], "ITERATIONS", world_rank);
    }
    if (world_size != 2) {
        if (world_rank == 0) {
            fprintf(stderr, "world size must be two; got %d\n", world_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Win_allocate(
        (MPI_Aint)sizeof(*window_base),
        (int)sizeof(*window_base),
        MPI_INFO_NULL,
        MPI_COMM_WORLD,
        &window_base,
        &window);

    MPI_Win_get_attr(window, MPI_WIN_MODEL, &window_model, &model_found);
    if (!model_found || *window_model != MPI_WIN_UNIFIED) {
        if (world_rank == 0) {
            fputs("this reproducer requires the unified MPI window model\n", stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    initial = (double)world_rank;
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, world_rank, 0, window);
    MPI_Put(&initial, 1, MPI_DOUBLE, world_rank, 0, 1, MPI_DOUBLE, window);
    MPI_Win_unlock(world_rank, window);

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf(
            "world=%d iterations=%d count=1\n",
            world_size,
            iterations);
        fflush(stdout);
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (world_rank == 1) {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, window);
            MPI_Get(&received, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, window);
            MPI_Win_unlock(0, window);
            printf("completed %d iterations\n", iteration + 1);
            fflush(stdout);
        }
    }

    /* Keep the target and its window alive until every Get epoch has ended. */
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_free(&window);
    if (world_rank == 0) {
        puts("PASS");
    }
    MPI_Finalize();
    return 0;
}
