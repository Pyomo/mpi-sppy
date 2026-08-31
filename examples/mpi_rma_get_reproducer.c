/*
 * Minimal C translation of mpi4py_xhat_collective_reproducer.py --mode
 * get-only.  It creates independent four-rank communicators and repeatedly
 * gets data from rank 0 of each communicator on rank 3.
 *
 * Compile:
 *   mpicc -O2 -Wall -Wextra -o mpi_rma_get_reproducer \
 *       examples/mpi_rma_get_reproducer.c
 *
 * Run with the same repeated 3+1 host placement as the Python reproducer:
 *   SLURM_HOSTFILE="$hostfile" srun -u -n 24 --distribution=arbitrary \
 *       ./mpi_rma_get_reproducer 100
 *
 * Arguments are ITERATIONS and GROUP_SIZE, both optional. GROUP_SIZE defaults
 * to four, with its last rank performing a one-double MPI_Get from rank zero.
 */

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    DEFAULT_ITERATIONS = 100,
    DEFAULT_GROUP_SIZE = 4
};

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
    int group;
    int group_rank;
    int iterations = DEFAULT_ITERATIONS;
    int group_size = DEFAULT_GROUP_SIZE;
    int reader_rank;
    MPI_Comm group_comm = MPI_COMM_NULL;
    MPI_Win window = MPI_WIN_NULL;
    double *window_base = NULL;
    double received;
    int *window_model = NULL;
    int model_found = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc > 3) {
        if (world_rank == 0) {
            fprintf(stderr, "usage: %s [ITERATIONS [GROUP_SIZE]]\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (argc >= 2) {
        iterations = parse_positive_int(argv[1], "ITERATIONS", world_rank);
    }
    if (argc == 3) {
        group_size = parse_positive_int(argv[2], "GROUP_SIZE", world_rank);
    }
    if (group_size < 2) {
        if (world_rank == 0) {
            fputs("GROUP_SIZE must be at least two\n", stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (world_size % group_size != 0) {
        if (world_rank == 0) {
            fprintf(stderr, "world size must be a multiple of %d\n", group_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    group = world_rank / group_size;
    group_rank = world_rank % group_size;
    reader_rank = group_size - 1;

    MPI_Comm_split(MPI_COMM_WORLD, group, world_rank, &group_comm);

    MPI_Win_allocate(
        (MPI_Aint)sizeof(*window_base),
        (int)sizeof(*window_base),
        MPI_INFO_NULL,
        group_comm,
        &window_base,
        &window);

    MPI_Win_get_attr(window, MPI_WIN_MODEL, &window_model, &model_found);
    if (!model_found || *window_model != MPI_WIN_UNIFIED) {
        if (world_rank == 0) {
            fputs("this reproducer requires the unified MPI window model\n", stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    *window_base = (double)world_rank;

    MPI_Barrier(group_comm);

    if (world_rank == 0) {
        printf(
            "world=%d groups=%d group_size=%d iterations=%d count=1\n",
            world_size,
            world_size / group_size,
            group_size,
            iterations);
        fflush(stdout);
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (group_rank == reader_rank) {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, window);
            MPI_Get(&received, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, window);
            MPI_Win_unlock(0, window);
        }

        MPI_Barrier(group_comm);

        if (group_rank == reader_rank) {
            printf("group=%d completed %d iterations\n", group, iteration + 1);
            fflush(stdout);
        }
    }

    MPI_Win_free(&window);
    MPI_Comm_free(&group_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        puts("PASS");
    }
    MPI_Finalize();
    return 0;
}
