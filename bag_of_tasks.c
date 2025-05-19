#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compile com: mpicc -o bag_of_tasks bag_of_tasks.c
// Execute com: mpirun --allow-run-as-root -np <P> ./bag_of_tasks <modo_send> <modo_recv>
// <modo_send> ∈ {0:MPI_Send, 1:MPI_Isend, 2:MPI_Rsend, 3:MPI_Bsend, 4:MPI_Ssend}
// <modo_recv> ∈ {0:MPI_Recv, 1:MPI_Irecv}

enum { SEND_SYNC=0, SEND_ISEND, SEND_RSEND, SEND_BSEND, SEND_SSEND };
enum { RECV_SYNC=0, RECV_IRECV };

static int is_prime(int n) {
    if (n < 2) return 0;
    int lim = (int)sqrt(n);
    for (int i = 2; i <= lim; ++i)
        if (n % i == 0) return 0;
    return 1;
}

static void send_int(int *buf, int dst, int tag, int mode) {
    MPI_Request req;
    switch (mode) {
        case SEND_SYNC:  MPI_Send(buf, 1, MPI_INT, dst, tag, MPI_COMM_WORLD); break; 
        case SEND_ISEND: MPI_Isend(buf, 1, MPI_INT, dst, tag, MPI_COMM_WORLD, &req);
                         MPI_Wait(&req, MPI_STATUS_IGNORE); break;
        case SEND_RSEND: MPI_Rsend(buf, 1, MPI_INT, dst, tag, MPI_COMM_WORLD); break;
        case SEND_BSEND: MPI_Bsend(buf, 1, MPI_INT, dst, tag, MPI_COMM_WORLD); break;
        case SEND_SSEND: MPI_Ssend(buf, 1, MPI_INT, dst, tag, MPI_COMM_WORLD); break;
    }
}

static void recv_int(int *buf, int src, int tag, int mode, MPI_Status *st) {
    MPI_Request req;
    switch (mode) {
        case RECV_SYNC:  MPI_Recv(buf, 1, MPI_INT, src, tag, MPI_COMM_WORLD, st); break;
        case RECV_IRECV: MPI_Irecv(buf, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &req);
                         MPI_Wait(&req, st); break;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            printf("Uso: %s <modo_send 0-4> <modo_recv 0-1>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int send_mode = atoi(argv[1]);
    int recv_mode = atoi(argv[2]);
    const int N = 100000;
    const int task_sz = 1000;

    // Aloca buffer para Bsend se necessário
    void *bsend_buffer = NULL;
    if (send_mode == SEND_BSEND) {
        int bsize;
        MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &bsize);
        bsize += MPI_BSEND_OVERHEAD;
        bsend_buffer = malloc(bsize);
        MPI_Buffer_attach(bsend_buffer, bsize);
    }

    int local_count = 0;
    double t_start = MPI_Wtime();

    if (size == 1) {
        for (int i = 0; i < N; ++i)
            if (is_prime(i)) local_count++;
        double t_end = MPI_Wtime();
        printf("Total primes found: %d\n", local_count);
        printf("Elapsed time: %.9f seconds\n", t_end - t_start);

        if (bsend_buffer) {
            int detached_size;
            MPI_Buffer_detach(&bsend_buffer, &detached_size);
            free(bsend_buffer);
        }

        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        int next_task = 0, done_workers = 0, buf;
        MPI_Status st;
        while (done_workers < size - 1) {
            recv_int(&buf, MPI_ANY_SOURCE, MPI_ANY_TAG, recv_mode, &st);
            if (buf == 1) {
                if (next_task >= N) {
                    buf = -1;
                } else {
                    buf = next_task;
                    next_task += task_sz;
                }
                send_int(&buf, st.MPI_SOURCE, 0, send_mode);
            } else if (buf == 2) {
                done_workers++;
            }
        }
    } else {
        MPI_Status st;
        int buf;
        while (1) {
            buf = 1; 
            send_int(&buf, 0, 1, send_mode);
            recv_int(&buf, 0, 0, recv_mode, &st);
            if (buf < 0) break;

            int start = buf; 
            int end = (buf + task_sz > N) ? N : buf + task_sz;
            for (int i = start; i < end; ++i)
                if (is_prime(i)) local_count++;
        }
        buf = 2;
        send_int(&buf, 0, 2, send_mode);
    }

    double t_end = MPI_Wtime();
    double t_total = t_end - t_start;

    int total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_max;
    MPI_Reduce(&t_total, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total primes found: %d\n", total_count);
        printf("Max elapsed time (wall-clock): %.9f seconds\n", t_max);
    }

    if (bsend_buffer) {
        int detached_size;
        MPI_Buffer_detach(&bsend_buffer, &detached_size);
        free(bsend_buffer);
    }

    MPI_Finalize();
    return 0;
}
