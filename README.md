# GMM_EM_Algorithm
Parallel implementation of the Expectation-Maximization Algorithm for the Gaussian Mixture Model in C

In holzer/EMAlgorithm, everything is pre-compiled and executed. To replicate, do the following:

To compile OpenMP:

gcc -g  gmm_em_openmp.c gmm_em_functions.c -o bin/gmm_em_openmp -fopenmp  -Wall -lm

To compile MPI:

mpicc -g gmm_em_mpi.c gmm_em_functions.c -o bin/gmm_em_mpi -Wall -lm

To run the script:

qsub batch.em

It will take ~40 walltime in the queue.

The labels will be in labels.txt and the output of the batch script will be in out.txt.
