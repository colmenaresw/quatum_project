#!/bin/bash
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=2
#SBATCH --partition=compute2011
#SBATCH --exclusive
date
for i in {1..48}
do
    echo "running for $i"
    mpirun -n $i python3.8 main_p.py
done
date