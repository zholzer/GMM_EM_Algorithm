#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <math.h>
#include <time.h>
#include <omp.h>

// reference https://ieeexplore.ieee.org/document/6787289
// data, command line
void initializeMeans(double X, int N, int d, int K, double mu);
void initializeCovariances(double X, int N, int d, double sigma);
void initializeCoefficients(int K, double alpha);

void EStep(double X, int N, int d, double mu, double sigma, double alpha, int K, double H);
void MStep0(double X, int N, int d, double H, int K, double mu, double alpha);
void MStep1(double X, int N, int d, double H, int K, double mu, double alpha, double sigma);
void checkConvergence(double H, int N, int K, double HPrev);
void getLabels(double H, int N, int K, int labelIndices); // labelIndices 1xN
// H is posterior probabuility, the output

void printMatrix(int n, int m, double x[n][m]);

int main(int argc, char *argv[]){
    // steps

    // 0. make data (in python)
 
    // 1. initialize 
    int N = 0, d = 1, K, thread_count, i, j; // initialize
    double index;
    
    char *fileName = argv[1]; // save pointer to command line args
    K = atoi(argv[2]);
    thread_count = atoi(argv[3]);

    FILE *fp = fopen(fileName,"r");
    // https://stackoverflow.com/questions/12733105/c-function-that-counts-lines-in-file
   char ch;
    while(!feof(fp)){
    ch = fgetc(fp);
        if(ch == '\n'){
            N++;
        }
    } 
    // https://www.programiz.com/c-programming/examples/read-file
    rewind(fp);
    while(!feof(fp)){
        if(ch == '\n'){break;}
        ch = fgetc(fp);
        if(ch == ','){
            d++;
        }
    }

    //https://stackoverflow.com/questions/36890624/malloc-a-2d-array-in-c
    double (*X)[d] = malloc(sizeof(int[N][d]));
    double (*H)[K] = malloc(sizeof(int[N][K]));
    double (*HPrev)[K] = malloc(sizeof(int[N][K]));

    double *mu = malloc(K*sizeof(double));
    double *sigma = malloc(K*sizeof(double));
    double *alpha = malloc(K*sizeof(double));
    double *labelIndices = malloc(N*sizeof(double));

    rewind(fp);
    char buffer[160];
    for (i = 0; i < N; i++){
        if (!fgets(buffer, 160, fp)) {printf("Incorrect dimensions. Something is wrong.\n"); break;}
        // If you need all the values in a row
        char *token;
        token = strtok(buffer, ",");
            for (j = 0; j < d; j++){
            // Just printing each integer here but handle as needed
            double n = atof(token);
            //printf("%lf\n", n);
            X[i][j] = n;
            //printf("%d %d %lf\n", i, j, X[i][j]);

            token = strtok(NULL, ",");
        }
    }
    //}
    // https://stackoverflow.com/questions/61078280/how-to-read-a-csv-file-in-c

    printMatrix(N, d, X);

    // input in command line csv filename, input number of components, and number threads
    // also wether they want labels
    // data is csv delimeted, Nxd (array length x dimension number)
    // thread_count is number of threads

    // use random points for means
    // use whole dataset covariance to start (could add regularization matrix *number components)
    // use uniform mixing coefficients as 1/# cluster

 void initializeMeans();
    void initializeCovariances();
    void initializeCoefficients();

    // 2. E-Step

    omp_set_num_threads(thread_count);
    # pragma omp parallel

    void EStep();
    //  a. split threads over array

    //  b. calculate expectations

    //  c. combine likelihood data



    // 3. M-Step
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
    # pragma omp single 

    return 0;
}

void initializeMeans(double X, int N, int d, int K, double mu){
    
}
void initializeCovariances(double X, int N, int d, double sigma);
void initializeCoefficients(int K, double alpha);
void EStep(double X, int N, int d, double mu, double sigma, double alpha, int K, double H);
void MStep0(double X, int N, int d, double H, int K, double mu, double alpha);
void MStep1(double X, int N, int d, double H, int K, double mu, double alpha, double sigma);
void checkConvergence(double H, int N, int K, double HPrev);
void getLabels(double H, int N, int K, int labelIndices);

void printMatrix(int n, int m, double x[n][m]){ // prints solution vector
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
        printf("%lf ", x[i][j]);
        }
        printf("\n");
    }
}
