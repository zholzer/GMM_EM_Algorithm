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

mpirun --oversubscribe -np 8 ./gmm_em_mpi dataShort.csv 3

echo -e ""
echo -e ""

processNum=(1 2 4 8 16 32)  # script to calculate speedup/efficiencies 
for process in "${processNum[@]}"
do
    echo "Processors: $process"
    t=$(mpirun --oversubscribe -np $process ./gmm_em_mpi dataShort.csv 3 | tail -n 1)
    if [ "$process" -eq 1 ]; then
        serialT=$t
        echo "The serial time is $t."
    else
        speedUp=$(echo "scale=2; $serialT / $t" | bc)
        efficiency=$(echo "scale=2; $serialT / ($process * $t)" | bc)
        echo "Speedup is $speedUp and efficiency is $efficiency."
    fi
done
echo -e "" 
