#!/bin/sh
#PBS -V
#PBS -l nodes="node13"
##PBS -l nodes=2:ppn=4:mpi
#PBS -N em
#PBS -j oe
#PBS -o out.txt
#PBS -q batch

cd $PBS_O_WORKDIR
echo "batch.em: running MonteCarlo test..."
cat $PBS_NODEFILE
echo "application output follows..."

export OMPI_MCA_btl=^openib

mpiexec -n 4 ./gmm_em_mpi dataShort.csv 3
