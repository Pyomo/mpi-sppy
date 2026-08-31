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
 *       ./mpi_rma_get_reproducer 100 264
 *
 * Arguments are ITERATIONS and COUNT, both optional. COUNT is the number of
 * doubles transferred by MPI_Get; 264 matches the padded Python buffer for a
 * payload length of 256.
 */

#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    GROUP_SIZE = 4,
    READER_RANK = 3,
    DEFAULT_ITERATIONS = 100,
    DEFAULT_COUNT = 264
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
    int provided;
    int world_rank;
    int world_size;
    int group;
    int group_rank;
    int iterations = DEFAULT_ITERATIONS;
    int count = DEFAULT_COUNT;
    MPI_Comm xhat_comm = MPI_COMM_NULL;
    MPI_Comm group_comm = MPI_COMM_NULL;
    MPI_Win window = MPI_WIN_NULL;
    double *window_base = NULL;
    double *received = NULL;
    int *window_model = NULL;
    int model_found = 0;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc > 3) {
        if (world_rank == 0) {
            fprintf(stderr, "usage: %s [ITERATIONS [COUNT]]\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (argc >= 2) {
        iterations = parse_positive_int(argv[1], "ITERATIONS", world_rank);
    }
    if (argc == 3) {
        count = parse_positive_int(argv[2], "COUNT", world_rank);
    }
    if (world_size % GROUP_SIZE != 0) {
        if (world_rank == 0) {
            fprintf(stderr, "world size must be a multiple of %d\n", GROUP_SIZE);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    group = world_rank / GROUP_SIZE;
    group_rank = world_rank % GROUP_SIZE;

    /* Keep the otherwise-unused Xhat communicator for a faithful first port. */
    MPI_Comm_split(
        MPI_COMM_WORLD,
        group_rank == READER_RANK ? 0 : MPI_UNDEFINED,
        world_rank,
        &xhat_comm);
    MPI_Comm_split(MPI_COMM_WORLD, group, world_rank, &group_comm);

    MPI_Win_allocate(
        (MPI_Aint)count * (MPI_Aint)sizeof(*window_base),
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

    for (int i = 0; i < count; ++i) {
        window_base[i] = (double)i;
    }
    if (group_rank == READER_RANK) {
        received = malloc((size_t)count * sizeof(*received));
        if (received == NULL) {
            fprintf(stderr, "rank %d: unable to allocate receive buffer\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    MPI_Barrier(group_comm);

    if (world_rank == 0) {
        printf(
            "world=%d groups=%d iterations=%d count=%d "
            "requested_thread=%d provided_thread=%d\n",
            world_size,
            world_size / GROUP_SIZE,
            iterations,
            count,
            MPI_THREAD_MULTIPLE,
            provided);
        fflush(stdout);
    }

    for (int iteration = 0; iteration < iterations; ++iteration) {
        MPI_Barrier(group_comm);

        if (group_rank == READER_RANK) {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, window);
            MPI_Get(received, count, MPI_DOUBLE, 0, 0, count, MPI_DOUBLE, window);
            MPI_Win_unlock(0, window);
        }

        MPI_Barrier(group_comm);

        if (group_rank == READER_RANK) {
            printf("group=%d completed %d iterations\n", group, iteration + 1);
            fflush(stdout);
        }
    }

    free(received);
    MPI_Win_free(&window);
    MPI_Comm_free(&group_comm);
    if (xhat_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&xhat_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        puts("PASS");
    }
    MPI_Finalize();
    return 0;
}
