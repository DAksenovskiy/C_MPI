#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
//cd source\repos\ConsoleApplication1\x64\Debug
//mpiexec -n 2 .\ConsoleApplication1.exe

float vander_polynomial(float x0, float v0, float mu, float A, float w, int t[], int n)
{
    float dydt;
    float cosf(float a);
    dydt = mu * (1 - pow(x0, 2)) * v0 - x0 - A * cosf(w * t[n]);
    if (n == 0)
    {
        dydt = v0;
    }
    return dydt;
}

int main(float *argc, char **argv)
{
    float x0 = -10.0;
    float v0 = 1.0;
    float mu = 1.0;
    float A = 20.0;
    float w = 12.0;
    float t[9000] = { 0 };
    float t1[9000] = { 0 };
    float t2[4500] = { 0 };
    float t3[4500] = { 0 };

    for (int i = 0; i < 1000; i++) {
        t[i] = i;
    }

    for (int i = 0; i < 30; i++) {
       // printf("%.3f", vander_polynomial(x0, v0, mu, A, w, t, i));
       // printf(" %.3f", vander_polynomial(0, 0, mu, A, w, t, i));
       // printf("\n");
    }
    MPI_Init(argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    if (rank == 1)
    {
        for (int i = 0; i < 4500; i++) {
            t2[i] = vander_polynomial(x0, v0, mu, A, w, t, i+12);
        }
        MPI_Send(&t2, 4500, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 0) {
        for (int i = 0; i < 4500; i++) {
            t1[i] = vander_polynomial(x0, v0, mu, A, w, t, i);
        }
        MPI_Status status;
        MPI_Recv(&t3, 4500, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < 12; i++) {
            printf("%.3f", t1[i]);
            printf(" %.3f", t3[i]);
            printf("\n");
        }
        end = MPI_Wtime();
        printf("\nRuntime = %f\n", end - start);
    }
    MPI_Finalize();
    return 0;
}
