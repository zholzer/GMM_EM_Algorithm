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

./bin/gmm_em_openmp data/data_2.csv 5 8 # shows results of smallest dataset
echo ""

export OMPI_MCA_btl=^openib

echo "batch.em: running EM test on the MPI program with data_2.csv..."

mpirun --oversubscribe -np 8 ./bin/gmm_em_mpi data/data_2.csv 5

echo ""
echo "batch.em: Begin table."

n_num=(2 3 4 5 6)
processNum=(1 2 4 8 16 32)  # script to calculate speedup/efficiencies 

for n in "${n_num[@]}"; 
do 
    echo ""
    echo "data_$n.csv"
    printf "%-7s%-13s%-13s%-9s%-9s%-9s%-8s\n" "#p" "t OMP" "t MPI" "S OMP" "S MPI" "E OMP" "E MPI"
    for process in "${processNum[@]}"
    do
        tOMP=$(./bin/gmm_em_openmp data/data_$n.csv 5 $process | tail -n 1)
        tMPI=$(mpirun --oversubscribe -np $process ./bin/gmm_em_mpi data/data_$n.csv 5 | tail -n 1)
        if [ "$process" -eq 1 ]; then
            serialTOMP=$tOMP
            serialTMPI=$tMPI
            printf "%-7s%-13s%-13s%-9s%-9s%-9s%-8s\n" "1" "$serialTOMP" "$serialTMPI" "NA" "NA" "NA" "NA"
        else
            speedUpOMP=$(echo "scale=2; $serialTOMP / $tOMP" | bc)
            speedUpMPI=$(echo "scale=2; $serialTMPI / $tMPI" | bc)
            efficiencyOMP=$(echo "scale=2; $serialTOMP / ($process * $tOMP)" | bc)
            efficiencyMPI=$(echo "scale=2; $serialTMPI / ($process * $tMPI)" | bc)
            printf "%-7s%-13s%-13s%-9s%-9s%-9s%-8s\n" "$process" "$tOMP" "$tMPI" "$speedUpOMP" "$speedUpMPI" "$efficiencyOMP" "$efficiencyMPI"
        fi
    done
done
