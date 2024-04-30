#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "gmm_em.h"

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

// Function to compute the squared Euclidean distance between two vectors
double computeDistanceSquared(double *v1, double *v2, int d) {
    double distanceSquared = 0;
    for (int i = 0; i < d; i++) {
        double diff = v1[i] - v2[i];
        distanceSquared += diff * diff;
    }
    return distanceSquared;
}

// Function to initialize means using K-means++ method
void initializeMeansKMeansPlusPlus(int N, int d, int K, double X[N][d], double mu[K][d]) {
    // Choose the first centroid randomly
    int firstCentroidIndex = rand() % N;
    for (int i = 0; i < d; i++) {
        mu[0][i] = X[firstCentroidIndex][i];
    }

    // Initialize array to store distances to nearest centroids
    double *nearestDistances = (double *)malloc(N * sizeof(double));

    // Loop to select the rest of the centroids
    for (int k = 1; k < K; k++) {
        double totalDistance = 0.0;

        // Compute distances to nearest centroids for each point
        for (int n = 0; n < N; n++) {
            double minDistanceSquared = DBL_MAX;
            for (int j = 0; j < k; j++) {
                double distanceSquared = computeDistanceSquared(X[n], mu[j], d);
                minDistanceSquared = fmin(minDistanceSquared, distanceSquared);
            }
            nearestDistances[n] = minDistanceSquared;
            totalDistance += minDistanceSquared;
        }

        // Choose the next centroid with probability proportional to square distance
        double threshold = ((double)rand() / RAND_MAX) * totalDistance;
        double sum = 0.0;
        int nextCentroidIndex = 0;
        while (sum <= threshold && nextCentroidIndex < N) {
            sum += nearestDistances[nextCentroidIndex];
            nextCentroidIndex++;
        }
        nextCentroidIndex--;

        // Set the next centroid
        for (int i = 0; i < d; i++) {
            mu[k][i] = X[nextCentroidIndex][i];
        }
    }

    free(nearestDistances);
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

double checkConvergence(double *flag, double *logLIPrev, int N, int K, double H[N][K], double alpha[K]){
    double *LI = calloc(N, sizeof(double));
    double logLI = 0.0, epsilon = 0.000001;
    for (int i = 0; i < N; i++){
        for (int k = 0; k < K; k++){
            LI[i] += alpha[k]*H[i][k]; // get likelihood at each N
        }
        logLI += log(LI[i]); // take log-likelihood of array
    }
    if (fabs((logLI - *logLIPrev)/logLI) < epsilon){ // check convergence
        printf("It converged with log-likelihood %f.\n", logLI);
        *flag = 1.0; // if true tell main
        return *flag;
    }
    else{*logLIPrev = logLI; return *logLIPrev;} // if false prev value is updated
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
