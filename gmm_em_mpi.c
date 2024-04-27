#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>

// reference https://ieeexplore.ieee.org/document/6787289
void initializeMeans(int N, int d, int K, double X[N][d], double *meanVector,double mu[K][d]);
void findMeanVector(int N, int d, double X[N][d], double *meanVector);
void initializeCovariances(int N, int d, double X[N][d], int K, double *meanVector, double (*sigma)[d][d]);
void initializeCoefficients(int K, double *alpha);

double find_determinant(int n, double sigma_m[n][n]);
void gaussJordan(int n, double matrix[n][n], double inverse[n][n]);
double pdf(int d, double sigma_m[d][d], double mu_m[d], double x_i[d]);

void EStep(int d, int K, double x_i[d], double mu[K][d], double (*sigma)[d][d], double alpha[K], double H_i[K]);
void MStep(int firstIndex, int lastIndex, int N, int d, int K, double X[N][d], double H[N][K], double mu[K][d], double alpha[K], double (*sigma)[d][d]);
void checkConvergence(int N, int K, double H[N][K], double alpha[K]);
int* getLabels(int N, int K, double H[N][K]); // labelIndices 1xN
// H is posterior probabuility, the output

void printMatrix(int n, int m, double x[n][m]);
void plotPoints(const char *filename, int N, int d, int K, double points[N][d], double H[N][K], double mu[K][d], int iter);

double logLIPrev;
double epsilon = 0.00000001;
int maxIter = 1000, flag = 0, my_rank, comm_sz;

int main(int argc, char *argv[]){
    // 0. make data (in python)
    double totalTime = 0.0;
    MPI_Init(&argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    double localStartTime = MPI_Wtime();
    //////// 1a. initialize variables, read in file, allocate memory ////////
    int N = 0, d = 1, K, i, j, firstIndex = 0, lastIndex = 0, quot, rem; // initialize

    // Read in command line inputs. csv file name, number of components, number of threads
    char *fileName = argv[1]; // save pointer to command line args
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

    // close file
    fclose(fp);

    //////// 1b. initialization for EM algorithm ////////
    // use random points for means
    // use whole dataset covariance to start (could add regularization matrix *number components)
    // use uniform mixing coefficients as 1/# cluster
     if(my_rank==0){

        findMeanVector(N, d, X, meanVector);  // modifies the meanVector array in place. Input to initializeMeans and initializeCovariances.

        initializeMeans(N, d, K, X, meanVector, mu);
        // print the values of the initial means
        printf("initial means: \n");
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
                //index1[i] = lastIndex; 
                MPI_Send(&lastIndex, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
            ranges[dest] = lastIndex - firstIndex;
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
    }
 
    else if (my_rank!=0){  
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

        }*/

        //////// 2. E-Step ////////    
        for (int row = firstIndex; row <= lastIndex; row++){
            // compute values for each row
            EStep(d, K, X[row], mu, sigma, alpha, H[row]);
        }

        if (my_rank!=0){ 
            MPI_Send(H[firstIndex], (lastIndex-firstIndex+1)*K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);         
        }
        else if (my_rank==0){
            for (int dest = 1; dest < comm_sz; dest++){ 
                MPI_Recv(H[firstIndexVec[dest]], (ranges[dest]+1)*K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   
            }      
        }

        if (my_rank==0){ 
            for (int dest = 1; dest < comm_sz; dest++){
                MPI_Send(H, N*K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);  
            }       
        }
        else if (my_rank!=0){
            MPI_Recv(H, N*K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);       
        }

        // print debugging
        MStep(firstIndex, lastIndex, N, d, K, X, H, mu, alpha, sigma);

        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank==0){
            //4. check for convergence; iteration number and epsilon
            if (iter == maxIter){
                printf("Maximum iteration of %d reached. No convergence.\n", maxIter); 
                flag = 1;
            }
            checkConvergence(N, K, H, alpha);
            for(int dest = 1; dest < comm_sz; dest++){
                MPI_Send(&flag, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            }
            if (flag == 1){
                printf("Converged at iteration %d.\n", iter);
                break;
            }
       }
        else if(my_rank!=0){
            MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (flag == 1){break;}
        }
    }
    // 5. get labels using maximum probabuility of feature vector among components (optional)
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
    MPI_Reduce(&localTime, &totalTime , 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get the maximum
    if (my_rank==0){
        printf("Time is:\n");
        printf("%f", totalTime); // send total time to bash
    }

    // 7. generating graphs, output stuff
    // free memory
    free(alpha);
    free(mu);
    free(sigma);
    free(X);
    free(H);
    free(labelIndices);
    free(meanVector);
    MPI_Finalize();

    return 0;
}

/// END OF MAIN

void printMatrix(int n, int m, double x[n][m]){
    double sum;
    for (int i = 0; i < n; i++){
        sum = 0.0;
        printf("%3d: ", i);
        for (int j = 0; j < m; j++)
        {
            printf("%lf ", x[i][j]);
            sum += x[i][j];
        }
        printf("\n");
    }
}

// find the mean vector for all the inputs by arithmetic mean for each dimension. Used by initializeMeans and initializeCovariances.
void findMeanVector(int N, int d, double X[N][d], double *meanVector)
{
    for (int i = 0; i < d; i++)
    {
        double sum = 0;
        for (int j = 0; j < N; j++)
        {
            sum += X[j][i];
        }
        meanVector[i] = sum / N;
    }
}

// initialize means by small perturbations about the mean vector
void initializeMeans(int N, int d, int K, double X[N][d], double *meanVector,double mu[K][d]){
    // add small perturbations about the mean vector
    srand(time(NULL));
    double pert_scale = 0.5 * meanVector[0]; // set the scale of the perturbations to be 10% of the first element of the mean vector
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < d; j++)
        {
            double perturbation = -pert_scale + ((double)rand() / RAND_MAX) * 2 * pert_scale; // this will give random double between -pert_scale and pert_scale
            mu[i][j] = meanVector[j] + perturbation;
        }
    }
}

void initializeCoefficients(int K, double *alpha)
{
    double value = 1.0 / K; // the initial values of the coefficients will be uniform and add up to 1

    for (int i = 0; i < K; i++)
    {
        alpha[i] = value;
    }
}

void initializeCovariances(int N, int d, double X[N][d], int K, double *meanVector, double (*sigma)[d][d])
{
    // initialize the covariance matrix
    double cov_matrix[d][d];
    // fill in the diagonals with the variances
    for (int dim = 0; dim < d; dim++)
    {
        double sum_sq_diff = 0; // initialize sum of squared differences outside of loop
        for (int j = 0; j < N; j++)
        {
            sum_sq_diff += (X[j][dim] - meanVector[dim]) * (X[j][dim] - meanVector[dim]);
        }
        cov_matrix[dim][dim] = sum_sq_diff / ((double) N-1.0);
    }

    // fill in the off-diagonals with the covariances
    for (int i = 0; i < d; i++)
    {
        for (int j = i + 1; j < d; j++)
        {
            double sum_product = 0;
            for (int k = 0; k < N; k++)
            {
                sum_product += (X[k][i] - meanVector[i]) * (X[k][j] - meanVector[j]);
            }
            // fill in the upper and lower triangle with the covariance
            cov_matrix[i][j] = sum_product / (N-1.0);
            cov_matrix[j][i] = sum_product / (N-1.0);
        }
    }

    // Copy the covariance matrix into the sigma 3D matrix, which is K x d x d, so K number of covariance matrices of size d x d
    for (int k = 0; k < K; k++){
        for (int i = 0; i < d; i++){
            for (int j = 0; j < d; j++){
                sigma[k][i][j] = cov_matrix[i][j];
            }
        }
    }

}

// I need some sub-functions to call for the PDF function, which is itself needed for the E-step.
// Includes: determinant of a matrix, inverse of a matrix

// function to calculate determinant of a matrix of size d x d
// modified from https://stackoverflow.com/questions/41384020/c-program-to-calculate-the-determinant-of-a-nxn-matrix
double find_determinant(int n, double sigma_m[n][n]) {
    int i, j, k, factor = 1;
    double det = 0.0;
    double newm[n - 1][n - 1];

    // Base case: when matrix is a single element
    if (n == 1) return sigma_m[0][0];

    // Calculate determinant using recursion
    for (i = 0; i < n; i++) {
        // Create the new matrix
        for (j = 1; j < n; j++) {
            for (k = 0; k < n; k++) {
                if (k != i) {
                    newm[j - 1][k < i ? k : (k - 1)] = sigma_m[j][k];
                }
            }
        }
        det += factor * sigma_m[0][i] * find_determinant(n - 1, newm);
        factor *= -1;
    }

    return det;
}

void gaussJordan(int n, double matrix[n][n], double inverse[n][n]) {
    double temp;
    double identity[n][n], tempMatrix[n][n];

    // make a local tempMatrix copy of the input matrix. Otherwise the input matrix will become an identity matrix over the course of gaussJordan
    memcpy(tempMatrix, matrix, sizeof(tempMatrix));

    // Initialize identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                identity[i][j] = 1.0;
            else
                identity[i][j] = 0.0;
        }
    }

    // Applying Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        temp = tempMatrix[i][i];
        for (int j = 0; j < n; j++) {
            tempMatrix[i][j] /= temp;
            identity[i][j] /= temp;
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                temp = tempMatrix[k][i];
                for (int j = 0; j < n; j++) {
                    tempMatrix[k][j] -= tempMatrix[i][j] * temp;
                    identity[k][j] -= identity[i][j] * temp;
                }
            }
        }
    }

    // Copy the result to the inverse matrix 
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = identity[i][j];
        }
    }
}



double pdf(int d, double sigma_m[d][d], double mu_m[d], double x_i[d]) {
    double det = find_determinant(d, sigma_m);
    double part_1 = 1.0 / ( pow(2.0 * M_PI, d / 2.0) * sqrt(det));

    // (x_i - mu_m)
    double diff[d];
    for (int i = 0; i < d; i++) {
        diff[i] = x_i[i] - mu_m[i];
    }

    // initialize the inverse matrix, then fill the inverse matrix with the gaussJordan function to get the inverse of sigma_m
    double inverse_sigma[d][d]; 
    gaussJordan(d, sigma_m, inverse_sigma);

    // matrix vector multiplication. initialize result vector
    double vec[d];
    for (int row = 0; row < d; row++) {
        vec[row] = 0.0;
        for (int col = 0; col < d; col++) {
            vec[row] += inverse_sigma[row][col] * diff[col];
        }
    }

    // vector transpose and matrix multiplication. initialize the result value
    double val = 0.0;
    for (int i = 0; i < d; i++) {
        val += vec[i] * diff[i];
    }

    double part_2 = -0.5 * val;

    return part_1 * exp(part_2);
}

// input is a row vector (1xd), output is a row of component probabilities (1xK) for that row vector
void EStep(int d, int K, double x_i[d], double mu[K][d], double (*sigma)[d][d], double alpha[K], double H_i[K]){
    
    for (int m = 0; m < K; m++) {
        double numerator = alpha[m] * pdf(d, sigma[m], mu[m], x_i);
        double denominator = 0.0;
        for (int k = 0; k < K; k++) {
            denominator += alpha[k] * pdf(d, sigma[k], mu[k], x_i);
        }
        H_i[m] = numerator / denominator;
    }

}

void MStep(int firstIndex, int lastIndex, int N, int d, int K, double X[N][d], double H[N][K], double mu[K][d], double alpha[K], double (*sigma)[d][d]){
    double vi=0.0, sum=0.0, local_wi = 0.0, local_vi, local_sum;
    double *wi = calloc(K, sizeof(double));
    // calculate sum of H over N at each k
    for (int k = 0; k < K; k++){
        local_wi = 0.0;
        for (int i = firstIndex; i <= lastIndex; i++){
            local_wi += H[i][k];
        }
        MPI_Reduce(&local_wi, &wi[k], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    for (int k = 0; k < K; k++){
        for (int j = 0; j < d; j++){
            // calculate sum of H*x over N at each k
            local_vi = 0.0;
            for (int i = firstIndex; i <= lastIndex; i++){
                local_vi += H[i][k]*X[i][j];
            }
            MPI_Reduce(&local_vi, &vi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (my_rank==0){mu[k][j] = vi/wi[k];} // update mu at each k and d
        }
        if (my_rank==0){alpha[k] = (1.0/(double) N)*wi[k];} // update alpha
    }

    if (my_rank==0){  
        for(int dest = 1; dest < comm_sz; dest++){
            MPI_Send(mu, K*d, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(alpha, K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Send(wi, K, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    }
    else if (my_rank!=0){  
        MPI_Recv(mu, K*d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(alpha, K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(wi, K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int k = 0; k < K; k++){
        for (int j0 = 0; j0 < d; j0++){
            for (int j1 = 0; j1 < d; j1++){
                local_sum = 0.0;
                for (int i = firstIndex; i <= lastIndex; i++){
                    local_sum += H[i][k] * (X[i][j0] - mu[k][j0]) * (X[i][j1] - mu[k][j1]); // all threads need to know H
                }
                MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (my_rank==0){sigma[k][j0][j1] = sum/wi[k];}
            }
        }
    }
    if (my_rank==0){  
        for(int dest = 1; dest < comm_sz; dest++){
            MPI_Send(sigma, K*d*d, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    }
    else if (my_rank!=0){  
        MPI_Recv(sigma, K*d*d, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    free(wi);
}

void checkConvergence(int N, int K, double H[N][K], double alpha[K]){
    double *LI = calloc(N, sizeof(double));
    double logLI = 0.0;
    for (int i = 0; i < N; i++){
        for (int k = 0; k < K; k++){
            LI[i] += alpha[k]*H[i][k]; // get likelihood at each N
        }
        logLI += log(LI[i]); // take log-likelihood of array
    }
    if (fabs((logLI - logLIPrev)/logLI) < epsilon){ // check convergence
        printf("It converged with log-likelihood %f.\n", logLI);
        flag = 1; // if true tell main
    }
    else{logLIPrev = logLI;} // if false prev value is updated
}

int* getLabels(int N, int K, double H[N][K]){
    int *labelIndices = calloc(N, sizeof(int));
    FILE *fpo;
    fpo = fopen("labelsMPI.txt", "w");
    for (int i = 0; i < N; i++){
        int maxIndex = 0;
        double maxVal = H[i][0];
        for (int k = 1; k < K; k++){
            if (H[i][k] > maxVal){
                maxVal = H[i][k];
                maxIndex = k;
            }
        }
        fprintf(fpo, "%d\n", maxIndex); //save to the txt file
        labelIndices[i] = maxIndex;  // save to array
    }
    return labelIndices; // output the array
}

void plotPoints(const char *filename, int N, int d, int K, double points[N][d], double H[N][K], double mu[K][d], int iter) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    // initialize colors array andfind the label indices from H array
    int *colors = calloc(N, sizeof(int));
    for (int i = 0; i < N; i++){
        int maxIndex = 0;
        double maxVal = H[i][0];
        for (int k = 1; k < K; k++){
            if (H[i][k] > maxVal){
                maxVal = H[i][k];
                maxIndex = k;
            }
        }
        colors[i] = maxIndex;  // save to array
    }

    // Write the data points to a temporary file
    for (int i = 0; i < N; i++) {
        fprintf(fp, "%f %f %d\n", points[i][0], points[i][1], colors[i]);
    }

    // write the mean vectors to the same file with a different color index
    for (int i = 0; i < K; i++) {
        fprintf(fp, "%f %f %d\n", mu[i][0], mu[i][1], i); // Offset color index for marker points
    }

    fclose(fp);

    // Plot using gnuplot
    FILE *gnuplotPipe = popen("gnuplot -persist", "w");
    if (gnuplotPipe == NULL) {
        perror("Error opening pipe to gnuplot");
        return;
    }

    // fprintf(gnuplotPipe, "set terminal qt font 'Arial,12'\n"); // for printing to screen
    fprintf(gnuplotPipe, "set terminal pngcairo\n"); // for saving plots as images
    fprintf(gnuplotPipe, "set output 'plots/plot_%d.png'\n", iter); // Save plots to directory
    fprintf(gnuplotPipe, "set title 'Labeled 2D Data Points: Iter = %d'\n", iter); // Title will have iter number
    fprintf(gnuplotPipe, "set xlabel 'X'\n");
    fprintf(gnuplotPipe, "set ylabel 'Y'\n");
    fprintf(gnuplotPipe, "set key autotitle columnheader\n"); // Enable automatic legend titles

    // Plot filled circles with palette coloring
    fprintf(gnuplotPipe, "set cbrange [0:2]\n");
    // Set the color palette
    fprintf(gnuplotPipe, "set palette color model RGB defined (0 'blue', 1 'red', 2 'black')\n");

    // Plot filled circles with palette coloring for the first N rows and with asterisks for the next K rows.
    fprintf(gnuplotPipe, "plot '%s' using 1:2:3 every ::0::%d-2 with points pt 7 palette pointsize 1 title 'Data Points', \
                    '' using 1:2:3 every ::%d-1::%d+%d-1 with points pt 3 palette pointsize 3 title 'Cluster Centers'\n", filename, N, N, N, K);

    fflush(gnuplotPipe);
    fprintf(gnuplotPipe, "exit\n");
    pclose(gnuplotPipe);
    free(colors);
}
