/*
 * Minimal C translation of mpi4py_xhat_collective_reproducer.py --mode
 * get-only. Two receiving ranks repeatedly get one double apiece from two
 * publishing ranks. Only the receivers synchronize on their cylinder
 * communicator before each Get.
 *
 * Compile:
 *   mpicc -O2 -Wall -Wextra -o mpi_rma_get_reproducer \
 *       examples/mpi_rma_get_reproducer.c
 *
 * Run one rank on each of four nodes:
 *   srun -u -N 4 -n 4 --ntasks-per-node=1 ./mpi_rma_get_reproducer 100
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
    int role;
    int pair;
    int strata_rank;
    int iterations = DEFAULT_ITERATIONS;
    MPI_Comm cylinder_comm = MPI_COMM_NULL;
    MPI_Comm strata_comm = MPI_COMM_NULL;
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
    if (world_size != 4) {
        if (world_rank == 0) {
            fprintf(stderr, "world size must be four; got %d\n", world_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    /* Ranks 0-1 publish; ranks 2-3 receive. Each pair is a strata comm. */
    role = world_rank / 2;
    pair = world_rank % 2;
    MPI_Comm_split(MPI_COMM_WORLD, role, pair, &cylinder_comm);
    MPI_Comm_split(MPI_COMM_WORLD, pair, role, &strata_comm);
    MPI_Comm_rank(strata_comm, &strata_rank);

    MPI_Win_allocate(
        (MPI_Aint)sizeof(*window_base),
        (int)sizeof(*window_base),
        MPI_INFO_NULL,
        strata_comm,
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
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, strata_rank, 0, window);
    MPI_Put(&initial, 1, MPI_DOUBLE, strata_rank, 0, 1, MPI_DOUBLE, window);
    MPI_Win_unlock(strata_rank, window);

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf(
            "world=%d iterations=%d count=1\n",
            world_size,
            iterations);
        fflush(stdout);
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (role == 1) {
            MPI_Barrier(cylinder_comm);
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, window);
            MPI_Get(&received, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, window);
            MPI_Win_unlock(0, window);
            printf("world_rank=%d completed %d iterations\n", world_rank, iteration + 1);
            fflush(stdout);
        }
    }

    /* Keep the target and its window alive until every Get epoch has ended. */
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_free(&window);
    MPI_Comm_free(&strata_comm);
    MPI_Comm_free(&cylinder_comm);
    if (world_rank == 0) {
        puts("PASS");
    }
    MPI_Finalize();
    return 0;
}
