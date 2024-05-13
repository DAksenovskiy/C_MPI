#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
#include <omp.h>

//cd source\repos\ConsoleApplication1\x64\Debug
//mpiexec -n 2 .\ConsoleApplication1.exe

float vander_polynomial(float x0, float v0, float mu, float A, float w, int t[], int n)
{

}

int main(float *argc, char **argv)
{

    int x[10000] = {0};
    int y[10000] = {0};
    int z[10000] = {0};
    float R[10000] = {0};
    float R1[5000] = {0};
    float R2[5000] = {0};

    //Центр галактики
    x[0] = 0;
    y[0] = 0;
    z[0] = 0;

    //Звезды
    for (int i = 1; i < 10000; i++)
    {
        x[i] = rand() % 2000;
        y[i] = rand() % 2000;
        z[i] = rand() % 2000;
    }
    
    MPI_Init(argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    //printf('%i', x[1]);
    if (rank == 1)
    {
        #pragma omp parallel
        {
            for (int i = 5000; i < 10000; i++)
            {
                while ((pow(x[0] - x[i], 2) + pow(y[0] - y[i], 2) + pow(z[0] - z[i], 2)) > R2[i - 5000] * R2[i - 5000])
                {
                    R2[i - 5000] = R2[i - 5000] + 1;
                }
            }
        }
        MPI_Send(&R2, 5000, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 0) {

        #pragma omp parallel
        {
            for (int i = 5000; i < 10000; i++)
            {
                while ((pow(x[0] - x[i], 2) + pow(y[0] - y[i], 2) + pow(z[0] - z[i], 2)) > R1[i] * R1[i])
                {
                    R1[i] = R1[i] + 1;
                }
            }
        }
        MPI_Status status;
        MPI_Recv(&R2, 5000, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);

        for (int i = 0; i < 5000; i++)
        {
            R[i] = R1[i];
        }
        for (int i = 5000; i < 10000; i++)
        {
            R[i] = R2[i-5000];
        }
        float Max = 0;
        for (int i = 0; i < 10000; i++)
        {
            if (R[i] > Max)
            {
                Max = R[i];
            }
        }
        end = MPI_Wtime();
        printf("\nMax = %f\n", Max);
        printf("\nRuntime = %f\n", end - start);
       // printf('%i', x[1]);

    }
    MPI_Finalize();
    return 0;
}