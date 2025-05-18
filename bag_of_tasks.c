#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compile com: mpicc -o naive_mpi naive_mpi.c
// Execute com: mpirun -np <P> ./naive_mpi <modo_send> <modo_recv>
// O primeiro argumento é o número de processos, o segundo o modo de envio (0 a 4) e o terceiro é o modo de recepção (0 ou 1).
// <modo_send> ∈ {0:MPI_Send, 1:MPI_Isend, 2:MPI_Rsend, 3:MPI_Bsend, 4:MPI_Ssend}
// <modo_recv> ∈ {0:MPI_Recv, 1:MPI_Irecv}

enum { SEND_SYNC=0, SEND_ISEND, SEND_RSEND, SEND_BSEND, SEND_SSEND };
enum { RECV_SYNC=0, RECV_IRECV };

static int is_prime(int n){
    if(n<2) return 0;
    int lim = (int)sqrt(n);
    for(int i=2;i<=lim;++i) if(n%i==0) return 0;
    return 1;
}

static void send_int(int *buf,int dst,int tag,int send_type){
    MPI_Request req;
    switch(send_type){
        case SEND_SYNC:  MPI_Send(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD); break;
        case SEND_ISEND: MPI_Isend(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD,&req);
                         MPI_Wait(&req,MPI_STATUS_IGNORE); break;
        case SEND_RSEND: MPI_Rsend(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
        case SEND_BSEND: MPI_Bsend(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
        case SEND_SSEND: MPI_Ssend(buf,1,MPI_INT,dst,tag,MPI_COMM_WORLD);    break;
    }
}

static void recv_int(int *buf,int src,int tag,int recv_type, MPI_Status *st){
    MPI_Request req;
    switch(recv_type){
        case RECV_SYNC:  MPI_Recv(buf,1,MPI_INT,src,tag,MPI_COMM_WORLD,st); break;
        case RECV_IRECV: MPI_Irecv(buf,1,MPI_INT,src,tag,MPI_COMM_WORLD,&req);
                         MPI_Wait(&req,st); break;
    }
}

int main(int argc,char **argv){
    double t_start, t_end, t_total;
    int rank, size, N=100000, task_sz=1000;
    int send_mode, recv_mode;
    if(argc<3){
        if(argc==1) printf("Uso: %s <modo_send 0–4> <modo_recv 0–1>\n",argv[0]);
        return 1;
    }
    send_mode = atoi(argv[1]);
    recv_mode = atoi(argv[2]);

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(size == 1){
        t_start = MPI_Wtime();
        int local_count = 0;
	for (int i = 0; i < N; ++i){
	    if (is_prime(i)) local_count++;
	}
	t_end = MPI_Wtime();
	printf("Total elapsed time: %.9f seconds\n", t_end - t_start);
	MPI_Finalize();
	return 0;
    }

    t_start = MPI_Wtime();

    if(rank==0){
        int next_task=0, done_workers=0, tag, buf;
        MPI_Status st;
        while(done_workers < size-1){
            recv_int(&buf,MPI_ANY_SOURCE,MPI_ANY_TAG,recv_mode, &st);
            tag = buf; // worker indica tipo de mensagem no próprio buf
            if(tag==1){  // pedido de tarefa
                if(next_task>=N){
                    buf = -1;
                    send_int(&buf,st.MPI_SOURCE,0,send_mode);
                } else {
                    buf = next_task;
                    send_int(&buf,st.MPI_SOURCE,0,send_mode);
                    next_task += task_sz;
                }
            } else if(tag==2){
                done_workers++;
            }
        }
    } else {
        MPI_Status st;
        int buf, local_count=0;
        while(1){
            buf = 1;
            send_int(&buf,0,1,send_mode);

            recv_int(&buf,0,0,recv_mode, &st);
            if(buf<0) break;

            int start = buf, end = buf+task_sz;
            if(end>N) end=N;
            for(int i=start;i<end;++i)
                if(is_prime(i)) local_count++;
        }
        buf = 2;
        send_int(&buf,0,2,send_mode);
    }

    t_end = MPI_Wtime();
    t_total = t_end - t_start;

    //double t_max;
    //MPI_Reduce(&t_total, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0){
        printf("Total elapsed time: %.9f seconds\n", t_total);
    }

    MPI_Finalize();
    return 0;
}
