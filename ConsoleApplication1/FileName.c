#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>       // for clock_t, clock(), CLOCKS_PER_SEC

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

int main(int *argc, char **argv)
{
    double time_spent = 0.0;
    float x0 = -1.0;
    float v0 = 1.0;
    float mu = 1.0;
    float A = 1.0;
    float w = 1.0;
    int t[90000] = {0};

    for (int i = 0; i < 90000; i++) {
        t[i] = i;
    }
    float k = 0;
  //  clock_t begin = clock();
  //  for (int i = 0; i < 90000; i++) {
  //      k = k + vander_polynomial(x0, v0, mu, A, w, t, i);
        // printf("%.3f", vander_polynomial(x0, v0, mu, A, w, t, i));
       //  printf("  %.3f", vander_polynomial(0, 0, mu, A, w, t, i));
       //  printf("\n");
 //   }
  //  clock_t end = clock();
  //  time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    //printf("The elapsed time is %f seconds\n", time_spent);


   // begin = clock();
    MPI_Init(argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double start, end;
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    start = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < 45000; i++) {
            k = k + vander_polynomial(x0, v0, mu, A, w, t, i);
          //  printf("  %.3f", vander_polynomial(0, 0, mu, A, w, t, i));
          //  printf("\n");
        }
        printf("\n%i", rank);
    }
    else if (rank == 1)
    {
        for (int i = 45000; i < 90000; i++) {
            k = k + vander_polynomial(x0, v0, mu, A, w, t, i);
           // printf("  %.3f", vander_polynomial(0, 0, mu, A, w, t, i));
           // printf("\n");
        }
        printf("\n%i", rank);
    }
    end = MPI_Wtime();
    MPI_Finalize();

    if (rank == 0) { /* use time on master node */
        printf("Runtime = %f\n", end - start);
    }
    //end = clock();
    //time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    //printf("\nThe elapsed time is %f seconds\n", time_spent);
    return 0;
}