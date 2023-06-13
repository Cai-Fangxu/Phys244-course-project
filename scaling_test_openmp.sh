#!/bin/bash

numbers=(128 64 32 16 8)

for num in "${numbers[@]}"; do
    str1="#SBATCH --output=\"output_log/neuron_openmp_"
    str2=".out\""
    line=$(grep -n "$str1.*$str2" slurm_neuron_openmp.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_openmp.sb

    str1="#SBATCH --cpus-per-task="
    str2=""
    line=$(grep -n "$str1.*$str2" slurm_neuron_openmp.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_openmp.sb

    str1="    ./run_openmp "
    str2=""
    line=$(grep -n "$str1.*$str2" slurm_neuron_openmp.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_openmp.sb

    sbatch slurm_neuron_openmp.sb
done
