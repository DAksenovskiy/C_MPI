#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
#include <omp.h>
#include <locale.h>


//cd source\repos\ConsoleApplication1\x64\Debug
//mpiexec -n 2 .\ConsoleApplication1.exe

float vander_polynomial(float x0, float v0, float mu, float A, float w, float T[], int n)
{
    float dydt;
    float cosf(float a);
    dydt = mu * (1 - pow(x0, 2)) * v0 - x0 - A * cosf(w * T[n]);
    if (n == 0)
    {
        dydt = v0;
    }
    return dydt;
}

int main(float* argc, char** argv)
{
    char* locale = setlocale(LC_ALL, "Russian");
  //  SetConsoleCP(1251);
  //  SetConsoleOutputCP(1251);
    float x0 = 1.1232233;
    float v0 = 0.2233123;
    float mu = 1.1232332;
    float A = -1.1222333;
    float w = -1.1111332;
    int n = 10000;
    float t[10000] = { 0 };
    float t1[10000] = { 0 };
    float t2[10000] = { 0 };
    float T[10000] = { 0 };

    for (int i = 0; i < n; i++)
    {
        T[i] = 30 * i / n;
    }
    omp_set_num_threads(4);
    MPI_Init(argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start, end;
    int k = n / size; //при size = 4 k = 2500

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for (int j = 1; j < size - 1; j++) {
        if (rank == j)
        {
            for (int i = k * j; i < k * j + 1; i++)
            {
                t1[i] = vander_polynomial(x0, v0, mu, A, w, T, i);
            }
            MPI_Send(&t1, 10000, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }
    if (rank == 0) {
        for (int i = 0; i < k; i++)
        {
            t1[i] = vander_polynomial(x0, v0, mu, A, w, T, i);
        }
        for (int i = 0; i < k; i++)
        {
            t[i] = t1[i];
        }
        MPI_Status status;
        for (int h = 1; h < size - 1; h++) {

            MPI_Recv(&t2, 10000, MPI_FLOAT, h, 0, MPI_COMM_WORLD, &status);
            for (int i = 0; i < k; i++)
            {
                t[i + k * h] = t2[i];
            }
        }
    }

        end = MPI_Wtime();
        if (rank == 0) {
            printf("\n\n\n\n");
            printf("\ntime = %f\n", end - start);
            printf("\n\n\n\n");
            // printf('%i', x[1]);
        }

    MPI_Finalize();
    return 0;
}
