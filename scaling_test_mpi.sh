#!/bin/bash

numbers=(128 64 32 16 8)

for num in "${numbers[@]}"; do
    str1="#SBATCH --output=\"output_log/neuron_mpi_"
    str2=".out\""
    line=$(grep -n "$str1.*$str2" slurm_neuron_mpi.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_mpi.sb

    str1="#SBATCH --ntasks-per-node="
    str2=""
    line=$(grep -n "$str1.*$str2" slurm_neuron_mpi.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_mpi.sb

    str1="    srun --mpi=pmi2 -n "
    str2=" --cpu-bind=rank ./run_mpi"
    line=$(grep -n "$str1.*$str2" slurm_neuron_mpi.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_mpi.sb

    sbatch slurm_neuron_mpi.sb
done

