/*
 * Minimal cross-node MPI_Get reproducer using one persistent lock-all epoch.
 * Ranks 0-1 expose one static double each. Ranks 2-3 repeatedly read from
 * their paired publisher, synchronizing with each other before every Get.
 *
 * Build from this directory with: make
 * Run one rank per node with:
 *   srun -u -N 4 -n 4 --ntasks-per-node=1 \
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

static void trace(int rank, int iteration, const char *stage)
{
    printf("TRACE rank=%d iteration=%d %s\n", rank, iteration, stage);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    int rank;
    int size;
    int role;
    int pair;
    int iterations = DEFAULT_ITERATIONS;
    int model_found = 0;
    int *window_model = NULL;
    MPI_Aint window_nbytes;
    MPI_Comm role_comm = MPI_COMM_NULL;
    MPI_Comm strata_comm = MPI_COMM_NULL;
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
    if (size != 4) {
        if (rank == 0) {
            fprintf(stderr, "world size must be four; got %d\n", size);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    role = rank / 2;
    pair = rank % 2;
    MPI_Comm_split(MPI_COMM_WORLD, role, pair, &role_comm);
    MPI_Comm_split(MPI_COMM_WORLD, pair, role, &strata_comm);

    window_nbytes = role == 0 ? (MPI_Aint)sizeof(*window_base) : 0;
    MPI_Win_allocate(window_nbytes, 1, MPI_INFO_NULL, strata_comm,
                     &window_base, &window);

    if (role == 0) {
        *window_base = 42.0 + rank;
    }

    MPI_Win_lock_all(0, window);
    MPI_Win_sync(window);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Get_processor_name(processor, &processor_length);
    printf("rank=%d host=%.*s\n", rank, processor_length, processor);
    fflush(stdout);

    MPI_Win_get_attr(window, MPI_WIN_MODEL, &window_model, &model_found);
    if (rank == 0) {
        printf("world=4 publishers=2 receivers=2 iterations=%d "
               "window_model=%s\n",
               iterations,
               model_found && *window_model == MPI_WIN_UNIFIED
                   ? "unified" : "separate-or-unknown");
        fflush(stdout);
    }

    if (role == 1) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            trace(rank, iteration, "before-barrier");
            MPI_Barrier(role_comm);
            trace(rank, iteration, "after-barrier");
            trace(rank, iteration, "before-get");
            MPI_Get(&received, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, window);
            trace(rank, iteration, "after-get");
            trace(rank, iteration, "before-flush");
            MPI_Win_flush(0, window);
            trace(rank, iteration, "after-flush");
            if (received != 42.0 + pair) {
                fprintf(stderr,
                        "rank=%d iteration=%d expected=%.17g actual=%.17g\n",
                        rank, iteration, 42.0 + pair, received);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_unlock_all(window);
    MPI_Win_free(&window);
    MPI_Comm_free(&strata_comm);
    MPI_Comm_free(&role_comm);

    if (rank == 0) {
        puts("PASS");
    }
    MPI_Finalize();
    return 0;
}
