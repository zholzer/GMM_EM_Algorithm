#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "gmm_em.h"
#include <omp.h>

// reference https://ieeexplore.ieee.org/document/6787289
void MStepOMP(int N, int d, int K, double X[N][d], double H[N][K], double mu[K][d], double alpha[K], double (*sigma)[d][d]);

int main(int argc, char *argv[]){
    // 0. make data (in python)
    double totalTime = 0.0, startTime, endTime;
    startTime = omp_get_wtime(); // start timing

    //////// 1a. initialize variables, read in file, allocate memory ////////
    int N = 0, d = 1, K, thread_count, i, j, maxIter = 100; // initialize
    double logLIPrev = 0.0, flag = 0.0;

    // Read in command line inputs. csv file name, number of components, number of threads
    // data is csv delimeted, Nxd (array length x dimension number)
    char *fileName = argv[1];
    K = atoi(argv[2]);
    thread_count = atoi(argv[3]);

    // Read in data from csv file, parse the dimension {d} and number of points {N} from the file.
    FILE *fp = fopen(fileName, "r");
    // https://stackoverflow.com/questions/12733105/c-function-that-counts-lines-in-file
    char ch;
    while (!feof(fp))
    {
        ch = fgetc(fp);
        if (ch == '\n')
        {
            N++;
        }
    }
    // https://www.programiz.com/c-programming/examples/read-file
    rewind(fp);
    while (!feof(fp))
    {
        if (ch == '\n')
        {
            break;
        }
        ch = fgetc(fp);
        if (ch == ',')
        {
            d++;
        }
    }

    // Allocate memory for X, H, HPrev, mu, sigma, alpha, labelIndices based on the dimension d and number of points N
    // https://stackoverflow.com/questions/36890624/malloc-a-2d-array-in-c
    double(*X)[d] = malloc(sizeof(double[N][d]));
    double(*H)[K] = malloc(sizeof(double[N][K]));
    double(*HPrev)[K] = malloc(sizeof(double[N][K]));
    double(*mu)[d] = malloc(d * sizeof(double[K][d]));

    double(*sigma)[d][d] = malloc(K * sizeof(double[d][d]));   // K number of matrices that are of size dxd

    double *alpha = malloc(K * sizeof(double));
    double *meanVector = malloc(d * sizeof(double));
    double *labelIndices = malloc(N * sizeof(double));

    // https://stackoverflow.com/questions/61078280/how-to-read-a-csv-file-in-c
    rewind(fp);
    char buffer[160];
    for (i = 0; i < N; i++)
    {
        if (!fgets(buffer, 160, fp))
        {
            printf("Incorrect dimensions. Something is wrong.\n");
            break;
        }
        char *token;
        token = strtok(buffer, ",");
        for (j = 0; j < d; j++)
        {
            double n = atof(token);
            X[i][j] = n;

            token = strtok(NULL, ",");
        }
    }
    fclose(fp); // close file

    //////// 1b. initialization for EM algorithm ////////
    // use k_means++ for means
    // use whole dataset covariance to start
    // use uniform mixing coefficients as 1/# cluster
    findMeanVector(N, d, X, meanVector);  // modifies the meanVector array in place. Input to initializeMeans and initializeCovariances.

    initializeMeansKMeansPlusPlus(N, d, K, X, mu);
    printf("initial means: \n"); // print the values of the initial means
    for (int i = 0; i < K; i++)
    {
        printf("vector_%d = [", i);
        for (int j = 0; j < d; j++)
        {
            printf("%lf, ", mu[i][j]);
        }
        printf("]\n");
    }

    initializeCoefficients(K, alpha);

    initializeCovariances(N, d, X, K, meanVector, sigma);

    omp_set_num_threads(thread_count);
    omp_set_schedule(omp_sched_guided, 0); // guided best at large size

    for (int iter = 1; iter <= maxIter; iter++){
        /*if ((iter -1) % 2 == 0){
            printf("iteration %d\n", iter - 1);
            //printMatrix(N, K, H);
            // printf("mu matrix:\n");
            // printMatrix(K, d, mu);
            plotPoints("plot.dat", N, d, K, X, H, mu, iter - 1);
            // printf("alpha vector: \n");
            // for (int i = 0; i < K; i++){
            //     printf("%lf\n", alpha[i]);
            // }

        }*/ // for plotting!

        //////// 2. E-Step ////////
    # pragma omp parallel for
        for (int row = 0; row < N; row++){ // compute values for each row
            EStep(d, K, X[row], mu, sigma, alpha, H[row]);
        }

        // 3. M-Step
        MStepOMP(N, d, K, X, H, mu, alpha, sigma);

        // 4. check for convergence; iteration number and epsilon
        if (iter == maxIter){
            printf("Maximum iteration of %d reached. No convergence.\n", maxIter); 
            return 0;
        }
        checkConvergence(&flag, &logLIPrev, N, K, H, alpha);
        if ((int) flag == 1){
        printf("Converged at iteration %d.\n", iter);
        break;
        }
    }

    // 5. get labels using maximum probabuility of feature vector among components
    printMatrix(K, d, mu);
    int *labels = getLabels(N, K, H);

    // try plotting with gnuplot
    // if (d == 2){
    //     plotPoints("plot.dat", N, d, K, X, H, mu, iter - 1);
    // } // for plotting

    // 6. implement timing
    endTime = omp_get_wtime(); // end timing
    totalTime = endTime - startTime;
    printf("Time is:\n");
    printf("%f", totalTime); // output to bash
    printf("\n");

    free(alpha); // free memory
    free(mu);
    free(sigma);
    free(X);
    free(H);
    free(HPrev);
    free(labelIndices);
    free(meanVector);
    free(labels);

    return 0;
}

void MStepOMP(int N, int d, int K, double X[N][d], double H[N][K], double mu[K][d], double alpha[K], double (*sigma)[d][d]){
    double vi, sum;
    double *wi = calloc(K, sizeof(double));

    for (int k = 0; k < K; k++){ // calculate sum of H over N at each k
    # pragma omp parallel for reduction(+: wi[k])
        for (int i = 0; i < N; i++){
            wi[k] += H[i][k];
        }
        for (int j = 0; j < d; j++){ // calculate sum of H*x over N at each k
            vi = 0.0;
            # pragma omp parallel for reduction(+: vi)
            for (int i = 0; i < N; i++){
                vi += H[i][k]*X[i][j];
            }
            mu[k][j] = vi/wi[k]; // update mu at each k and d
        }
        alpha[k] = (1.0/(double) N)*wi[k]; // update alpha
    }

    for (int k = 0; k < K; k++){
        for (int j0 = 0; j0 < d; j0++){
            for (int j1 = 0; j1 < d; j1++){
                sum = 0.0;
                # pragma omp parallel for reduction(+: sum)
                for (int i = 0; i < N; i++){
                    sum += H[i][k] * (X[i][j0] - mu[k][j0]) * (X[i][j1] - mu[k][j1]); // get numerator
                }
                sigma[k][j0][j1] = sum/wi[k]; // update sigma
            }
        }
    }

    free(wi);
}
