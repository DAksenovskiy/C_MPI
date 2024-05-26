#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC
#include <omp.h>
#include <locale.h>

// Для множества звезд заданных координатами в трехмерной декартовой системе определить минимальную ограничивающую сферу

//cd source\repos\ConsoleApplication1\x64\Debug
//mpiexec -n 2 .\ConsoleApplication1.exe

int main(float* argc, char** argv)
{
    char* locale = setlocale(LC_ALL, "Russian");
    //  SetConsoleCP(1251);
    //  SetConsoleOutputCP(1251);
    int x[10000] = { 0 };
    int y[10000] = { 0 };
    int z[10000] = { 0 };
    float R[10000] = { 0 };
    float R1[10000] = { 0 };
    float R2[10000] = { 0 };
    int n = 10000;
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
                while ((pow(x[0] - x[i], 2) + pow(y[0] - y[i], 2) + pow(z[0] - z[i], 2)) > R1[i] * R1[i])
                {
                    R1[i] = R1[i] + 1;
                }
            }
            MPI_Send(&R1, 10000, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        for (int i = 0; i < k; i++)
        {
            while ((pow(x[0] - x[i], 2) + pow(y[0] - y[i], 2) + pow(z[0] - z[i], 2)) > R1[i] * R1[i])
            {
                R1[i] = R1[i] + 1;
            }
        }
        for (int i = 0; i < k; i++)
        {
            R[i] = R1[i];
        }
        MPI_Status status;
        for (int j = 1; j < size - 1; j++) {

            MPI_Recv(&R2, 10000, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &status);
            for (int i = 0; i < k; i++)
            {
                R[i + k * j] = R2[i];
            }
        }
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
    if (rank == 0) {
        //printf("\nДля множества звезд заданных координатами в трехмерной декартовой системе определить минимальную ограничивающую сферу\n");
        printf("\n\n\n\n");
        printf("\nR = %f\n", Max); //R минимальной ограничивающей сферы
        printf("\ntime = %f\n", end - start);
        printf("\n\n\n\n");
        // printf('%i', x[1]);
    }

    MPI_Finalize();
    return 0;
}
