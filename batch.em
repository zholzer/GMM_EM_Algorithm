#!/bin/sh
## PBS batch directives:
#PBS -V
#PBS -l nodes=2:ppn=16:mpi     # this will request 1 node and the max 16 processors. 2 nodes not working currently.
#PBS -N em
#PBS -j oe
#PBS -o out.txt
#PBS -q batch

cd $PBS_O_WORKDIR

echo "batch.em: running EM test on the OpenMP program with data_2.csv..."

./gmm_em_openmp data_2.csv 5 8

n_num=(2 3 4 5 6)
processNum=(1 2 4 8 16 32)  # script to calculate speedup/efficiencies 

for n in "${n_num[@]}"; 
do 
    echo ""
    echo "data_$n.csv"
    for process in "${processNum[@]}"
    do
        t=$(./gmm_em_openmp data_$n.csv 5 $process | tail -n 1)
        if [ "$process" -eq 1 ]; then
            serialT=$t
            echo "serial time = $t "
        else
            speedUp=$(echo "scale=2; $serialT / $t" | bc)
            efficiency=$(echo "scale=2; $serialT / ($process * $t)" | bc)
            echo "np = $process, time = $t,speedup = $speedUp, efficiency = $efficiency."
        fi
    done
done

echo -e ""
echo "batch.em: running EM test on the MPI program with data_2.csv..."

export OMPI_MCA_btl=^openib

mpirun --oversubscribe -np 8 ./gmm_em_mpi data_2.csv 5

n_num=(2 3 4 5 6)
processNum=(1 2 4 8 16 32)  # script to calculate speedup/efficiencies 

for n in "${n_num[@]}"; 
do 
    echo ""
    echo "data_$n.csv"
    for process in "${processNum[@]}"
    do
        t=$(mpirun --oversubscribe -np $process ./gmm_em_mpi data_$n.csv 5 | tail -n 1)
        if [ "$process" -eq 1 ]; then
            serialT=$t
            echo "serial time = $t "
        else
            speedUp=$(echo "scale=2; $serialT / $t" | bc)
            efficiency=$(echo "scale=2; $serialT / ($process * $t)" | bc)
            echo "np = $process, time = $t,speedup = $speedUp, efficiency = $efficiency."
        fi
    done
done
