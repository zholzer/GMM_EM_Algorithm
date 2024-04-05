#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// reference https://ieeexplore.ieee.org/document/6787289
// data, command line
void initializeMeans(int N, int d, int K, double X[N][d], double *meanVector,double mu[K][d]);
void findMeanVector(int N, int d, double X[N][d], double *meanVector);
void initializeCovariances(int N, int d, double X[N][d], int K, double *meanVector, double (*sigma)[d][d]);
void initializeCoefficients(int K, double *alpha);

double find_determinant(int n, double sigma_m[n][n]);
void gaussJordan(int n, double matrix[n][n], double inverse[n][n]);
double pdf(int d, double sigma_m[d][d], double mu_m[d], double x_i[d]);

void EStep(int d, int K, double x_i[d], double mu[K][d], double (*sigma)[d][d], double alpha[K], double H_i[K]);
void MStep0(int N, int d, int K, double X[N][d], double H, double mu[K][d], double *alpha);
void MStep1(int N, int d, int K, double X[N][d], double H, double mu[K][d], double *alpha, double (*sigma)[d][d]);
void checkConvergence(int N, int K, double H, double HPrev);
void getLabels(int N, int K, double H, int labelIndices); // labelIndices 1xN
// H is posterior probabuility, the output

void printMatrix(int n, int m, double x[n][m]);

int main(int argc, char *argv[])
{
    // 0. make data (in python)

    //////// 1a. initialize variables, read in file, allocate memory ////////
    int N = 0, d = 1, K, thread_count, i, j; // initialize
    double index;

    // Read in command line inputs. csv file name, number of components, number of threads
    // also whether they want labels 
    // data is csv delimeted, Nxd (array length x dimension number)
    // thread_count is number of threads
    char *fileName = argv[1]; // save pointer to command line args
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

    // close file
    fclose(fp);

    // print debugging
    // printMatrix(N, d, X);

    //////// 1b. initialization for EM algorithm ////////
    // use random points for means
    // use whole dataset covariance to start (could add regularization matrix *number components)
    // use uniform mixing coefficients as 1/# cluster

    findMeanVector(N, d, X, meanVector);  // modifies the meanVector array in place. Input to initializeMeans and initializeCovariances.

    initializeMeans(N, d, K, X, meanVector, mu);
    // print the values of the initial means
    printf("initial means: \n");
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
    // print debugging
    // for (int i = 0; i < K; i++){
    //     printf("%lf\n", alpha[i]);
    // }

    initializeCovariances(N, d, X, K, meanVector, sigma);
    // print debugging
    for (int k = 0; k < K; k++){
        printf("covariance matrix %d: \n", k);
        for (int i = 0; i < d; i++){
            for (int j = 0; j < d; j++){
                printf("%.4f\t", sigma[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    //////// 2. E-Step ////////

    omp_set_num_threads(thread_count);
    # pragma omp parallel for
    for (int row = 0; row < N; row++){
        // compute values for each row
        EStep(d, K, X[row], mu, sigma, alpha, H[row]);
    }
    // print debugging
    printf("E-Step: \n");
    printMatrix(N, K, H);


    /* 3. M-Step
    void MStep0();
    //  a. create temp vectors

    //  b. parallelize with partial sums

    //  c. calculate alpha and mu

    void MStep1(); // yellow part becomes alpha*N
    //  d. create temp vector

    //  e. parallelize with partial sums

    //  f. calculate sigma

    // 4. check for convergence; iteration number and epsilon
    void checkConvergence();

    // 5. get labels using maximum probabuility of feature vector among components (optional)
    void getLabels(); // index+1 of maximum of each row

    // 6. implement timing

    // 7. generating graphs, output stuff
    # pragma omp single */

    // free memory
    free(alpha);
    free(mu);
    free(sigma);
    free(X);
    free(H);
    free(HPrev);
    free(labelIndices);
    free(meanVector);

    return 0;
}

void printMatrix(int n, int m, double x[n][m])
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%lf ", x[i][j]);
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
        // print debugging
        // printf("%i: %lf \n", i, meanVector[i]);
    }
}

// initialize means by small perturbations about the mean vector
void initializeMeans(int N, int d, int K, double X[N][d], double *meanVector,double mu[K][d])
{
    // add small perturbations about the mean vector
    srand(time(NULL));
    double pert_scale = 0.1 * meanVector[0]; // set the scale of the perturbations to be 10% of the first element of the mean vector
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < d; j++)
        {
            double perturbation = -pert_scale + ((double)rand() / RAND_MAX) * 2 * pert_scale; // this will give random double between -pert_scale and pert_scale
            mu[i][j] = meanVector[j] + perturbation;
            // print debugging
            // printf("%i: %lf \n", i, mu[i][j]);
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
        cov_matrix[dim][dim] = sum_sq_diff / (N-1);
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
            cov_matrix[i][j] = sum_product / (N-1);
            cov_matrix[j][i] = sum_product / (N-1);
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
    double det = 0;
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
    double identity[n][n];

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
        temp = matrix[i][i];
        for (int j = 0; j < n; j++) {
            matrix[i][j] /= temp;
            identity[i][j] /= temp;
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                temp = matrix[k][i];
                for (int j = 0; j < n; j++) {
                    matrix[k][j] -= matrix[i][j] * temp;
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
    double part_1 = 1 / ( pow(2 * M_PI, d / 2.0) * sqrt(det));

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
