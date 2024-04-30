#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "gmm_em.h"
#include <mpi.h>

int my_rank, comm_sz; // global for convenience
// reference https://ieeexplore.ieee.org/document/6787289
void MStepMPI(int firstIndex, int lastIndex, int N, int d, int K, double X[N][d], double H[N][K], double mu[K][d], double alpha[K], double (*sigma)[d][d]);

int main(int argc, char *argv[]){
    // 0. make data (in python)
    double totalTime = 0.0;
    MPI_Init(&argc , &argv); // initialize MPI stuff
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    double localStartTime = MPI_Wtime(); // start timing

    //////// 1a. initialize variables, read in file, allocate memory ////////
    int N = 0, d = 1, K, i, j, firstIndex = 0, lastIndex = 0, quot, rem, maxIter = 100; // initialize
    double logLIPrev = 0.0, flag = 0;

    // Read in command line inputs. csv file name, number of components, number of threads using mpiexec/mpirun
    char *fileName = argv[1]; 
    K = atoi(argv[2]);

    // Read in data from csv file, parse the dimension {d} and number of points {N} from the file.
    FILE *fp = fopen(fileName, "r");
    // https://stackoverflow.com/questions/12733105/c-function-that-counts-lines-in-file
    char ch;
    while (!feof(fp)){
        ch = fgetc(fp);
        if (ch == '\n')
        {
            N++;
        }
    }
    // https://www.programiz.com/c-programming/examples/read-file
    rewind(fp);
    while (!feof(fp)){
        if (ch == '\n'){
            break;
        }
        ch = fgetc(fp);
        if (ch == ','){
            d++;
        }
    }

    // Allocate memory for X, H, HPrev, mu, sigma, alpha, labelIndices based on the dimension d and number of points N
    // https://stackoverflow.com/questions/36890624/malloc-a-2d-array-in-c
    double(*X)[d] = malloc(sizeof(double[N][d]));
    double(*H)[K] = malloc(sizeof(double[N][K]));
    double(*mu)[d] = malloc(d * sizeof(double[K][d]));

    double(*sigma)[d][d] = malloc(K * sizeof(double[d][d]));   // K number of matrices that are of size dxd

    double *alpha = malloc(K * sizeof(double));
    double *meanVector = malloc(d * sizeof(double));
    double *labelIndices = malloc(N * sizeof(double));

    int* ranges = malloc(comm_sz * sizeof(int));
    int* firstIndexVec = malloc(comm_sz * sizeof(int));

    // https://stackoverflow.com/questions/61078280/how-to-read-a-csv-file-in-c
    rewind(fp);
    char buffer[160];
    for (i = 0; i < N; i++){
        if (!fgets(buffer, 160, fp)){
            printf("Incorrect dimensions. Something is wrong.\n");
            break;
        }
        char *token;
        token = strtok(buffer, ",");
        for (j = 0; j < d; j++){
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
     if(my_rank==0){
        
        findMeanVector(N, d, X, meanVector);  // modifies the meanVector array in place. Input to initializeMeans and initializeCovariances.
        initializeMeansKMeansPlusPlus(N, d, K, X, mu);
        printf("initial means: \n"); // print the values of the initial means
        for (int i = 0; i < K; i++){
            printf("vector_%d = [", i);
            for (int j = 0; j < d; j++)
            {
                printf("%lf, ", mu[i][j]);
            }
            printf("]\n");
        }

        initializeCoefficients(K, alpha);
        initializeCovariances(N, d, X, K, meanVector, sigma);

        quot = N/comm_sz; // quotient is length of indices per processor
        rem = N%comm_sz; // remainder is amount of processes to add an extra element to

        for (int dest = 0; dest<comm_sz; dest++){
            if (dest<rem){ // for the length of the remainder
                lastIndex = firstIndex + quot; // last index is first index plus length of index range
            }
            else{ // once no more remainder
                lastIndex = firstIndex + quot - 1; // processors should have one less element so they have roughly the same number of elements
            }

            if (dest!=0){
                MPI_Send(&firstIndex, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Send(&lastIndex, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
            ranges[dest] = lastIndex - firstIndex; // save range and first index for E-Step later
            firstIndexVec[dest] = firstIndex;
            firstIndex = lastIndex + 1; // first index always one more then previous last

            if (dest!=0){
                MPI_Send(mu, K*d, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
                MPI_Send(alpha, K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
                MPI_Send(sigma, K*d*d, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        }

        int dest = 0;
        firstIndex = 0;
        if (dest<rem){ // for the length of the remainder
            lastIndex = firstIndex + quot; // last index is first index plus length of index range
        }
        else{ // once no more remainder
            lastIndex = firstIndex + quot - 1; // processors should have one less element so they have roughly the same number of elements
        }
        ranges[dest] = lastIndex - firstIndex;
        firstIndexVec[dest] = firstIndex;
    }
 
    else if (my_rank!=0){  // other threads receieve their indexes and initial parameters
        MPI_Recv(&firstIndex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        MPI_Recv(&lastIndex, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(mu, K*d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(alpha, K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(sigma, K*d*d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int iter = 1; iter <= maxIter; iter++){
        /*if ((iter -1) % 10 == 0)
        {
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
        for (int row = firstIndex; row <= lastIndex; row++){ // compute values for each row
            EStep(d, K, X[row], mu, sigma, alpha, H[row]);
        }

        if (my_rank!=0){ // threads send their portion of H to main
            MPI_Send(H[firstIndex], (lastIndex-firstIndex+1)*K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);         
        }
        else if (my_rank==0){ // main now has all of H updated
            for (int dest = 1; dest < comm_sz; dest++){ 
                MPI_Recv(H[firstIndexVec[dest]], (ranges[dest]+1)*K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   
            }      
        }

        if (my_rank==0){ // main sends updated H for M-step
            for (int dest = 1; dest < comm_sz; dest++){
                MPI_Send(H, N*K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);  
            }       
        }
        else if (my_rank!=0){ // threads receive updated H for M-step
            MPI_Recv(H, N*K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);       
        }

        // 3. M-Step
        MStepMPI(firstIndex, lastIndex, N, d, K, X, H, mu, alpha, sigma);

        //4. check for convergence; iteration number and epsilon
        if (my_rank==0){ // main thread checks
            if (iter == maxIter){
                printf("Maximum iteration of %d reached. No convergence.\n", maxIter); 
                flag = 1.0; // update flag if max iterations reached
            }
            checkConvergence(&flag, &logLIPrev, N, K, H, alpha); // check if converged
            for(int dest = 1; dest < comm_sz; dest++){ // send flag to threads
                MPI_Send(&flag, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
            if ((int) flag == 1){ // break if converged
                printf("Converged at iteration %d.\n", iter);
                break;
            }
       }
        else if(my_rank!=0){ // other threads receive flag and break if it is 1
            MPI_Recv(&flag, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if ((int) flag == 1){break;}
        }
    }
    // 5. first thread get labels using maximum probabuility of feature vector among components
    if(my_rank==0){
        printMatrix(K,d,mu);
        int *labels = getLabels(N, K, H);
        free(labels);
    }
    // try plotting with gnuplot
    // if (d == 2){
    //     plotPoints("plot.dat", N, d, K, X, H, mu, iter - 1);

    // 6. implement timing
    double localEndTime  = MPI_Wtime(); // get ending time
    double localTime = localEndTime - localStartTime; // get total time
    MPI_Reduce(&localTime, &totalTime , 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get the maximum of threads
    if (my_rank==0){
        printf("Time is:\n");
        printf("%f", totalTime); // send total time to bash
        printf("\n");
    }

    free(alpha); // free memory
    free(mu);
    free(sigma);
    free(X);
    free(H);
    free(labelIndices);
    free(meanVector);
    MPI_Finalize();

    return 0;
}

// calculates the MStep with MPI, ouputs updated parameters
void MStepMPI(int firstIndex, int lastIndex, int N, int d, int K, double X[N][d], double H[N][K], double mu[K][d], double alpha[K], double (*sigma)[d][d]){
    double vi=0.0, sum=0.0, local_wi = 0.0, local_vi, local_sum;
    double *wi = calloc(K, sizeof(double));
    
    for (int k = 0; k < K; k++){ // calculate sum of H over N at each k
        local_wi = 0.0;
        for (int i = firstIndex; i <= lastIndex; i++){
            local_wi += H[i][k];
        }
        MPI_Reduce(&local_wi, &wi[k], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        for (int j = 0; j < d; j++){ // calculate sum of H*x over N at each k
            local_vi = 0.0;
            for (int i = firstIndex; i <= lastIndex; i++){
                local_vi += H[i][k]*X[i][j];
            }
            MPI_Reduce(&local_vi, &vi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (my_rank==0){mu[k][j] = vi/wi[k];} // update mu at each k and d
        }
        if (my_rank==0){alpha[k] = (1.0/(double) N)*wi[k];} // update alpha
    }

    if (my_rank==0){ // main sends updated parameters
        for(int dest = 1; dest < comm_sz; dest++){
            MPI_Send(mu, K*d, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(alpha, K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(wi, K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    }
    else if (my_rank!=0){  // threads recieve updated parameters
        MPI_Recv(mu, K*d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(alpha, K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(wi, K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int k = 0; k < K; k++){
        for (int j0 = 0; j0 < d; j0++){
            for (int j1 = 0; j1 < d; j1++){
                local_sum = 0.0;
                for (int i = firstIndex; i <= lastIndex; i++){ // calculate numerator
                    local_sum += H[i][k] * (X[i][j0] - mu[k][j0]) * (X[i][j1] - mu[k][j1]); 
                }
                MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (my_rank==0){sigma[k][j0][j1] = sum/wi[k];} // update sigma
            }
        }
    }
    
    if (my_rank==0){  // main sends updated parameters
        for(int dest = 1; dest < comm_sz; dest++){
            MPI_Send(sigma, K*d*d, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    }
    else if (my_rank!=0){  // threads recieve updated parameters
        MPI_Recv(sigma, K*d*d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    free(wi);
}
