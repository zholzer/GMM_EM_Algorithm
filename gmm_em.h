void initializeMeans(int N, int d, int K, double X[N][d], double *meanVector,double mu[K][d]);
void findMeanVector(int N, int d, double X[N][d], double *meanVector);
void initializeCovariances(int N, int d, double X[N][d], int K, double *meanVector, double (*sigma)[d][d]);
void initializeCoefficients(int K, double *alpha);

double computeDistanceSquared(double *v1, double *v2, int d);
void initializeMeansKMeansPlusPlus(int N, int d, int K, double X[N][d], double mu[K][d]);

double find_determinant(int n, double sigma_m[n][n]);
void gaussJordan(int n, double matrix[n][n], double inverse[n][n]);
double pdf(int d, double sigma_m[d][d], double mu_m[d], double x_i[d]);

void EStep(int d, int K, double x_i[d], double mu[K][d], double (*sigma)[d][d], double alpha[K], double H_i[K]);
double checkConvergence(double *flag, double *logLIPrev, int N, int K, double H[N][K], double alpha[K]);
int* getLabels(int N, int K, double H[N][K]); // labelIndices 1xN, H is posterior probabuility, the output
void printMatrix(int n, int m, double x[n][m]);
void plotPoints(const char *filename, int N, int d, int K, double points[N][d], double H[N][K], double mu[K][d], int iter);
