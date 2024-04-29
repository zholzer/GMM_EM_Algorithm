#!/bin/sh
## PBS batch directives:
#PBS -V
#PBS -l nodes=1:ppn=16:mpi     # this will request 1 node and the max 16 processors. 2 nodes not working currently.
#PBS -N EM
#PBS -j oe
#PBS -q batch

cd $PBS_O_WORKDIR
NCORES=`wc -w < $PBS_NODEFILE`
LOCATION=`pwd`
echo "LOCATION== $LOCATION"
echo "Running $NCORES processes from $(tail -n 1 "$PBS_NODEFILE")"
echo "Output:"

export OMPI_MCA_btl=^openib

n_num=(2 3 4 5 6)
processNum=(1 2 4 8 16 32)  # script to calculate speedup/efficiencies 

for n in "${n_num[@]}"; 
do 
    echo ""
    echo "data_$n.csv"
    for process in "${processNum[@]}"
    do
        t=$(mpirun -mca plm_rsh_agent rsh --map-by node -hostfile $PBS_NODEFILE --oversubscribe -np $process ./gmm_em_mpi data_$n.csv 5 | tail -n 1)
        if [ "$process" -eq 1 ]; then
            serialT=$t
            echo "serial time = $t"
        else
            speedUp=$(echo "scale=2; $serialT / $t" | bc)
            efficiency=$(echo "scale=2; $serialT / ($process * $t)" | bc)
            echo "np = $process, time = $t,speedup = $speedUp, efficiency = $efficiency."
        fi
    done
done