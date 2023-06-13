#!/bin/bash

num_thread=(64 32 16 8 4)
num_process=2

for num in "${num_thread[@]}"; do
    str1="#SBATCH --output=\"output_log/neuron_mpi_openmp_${num_process}_"
    str2=".out\""
    line=$(grep -n "$str1.*$str2" slurm_neuron_mpi_openmp.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_mpi_openmp.sb

    str1="#SBATCH --cpus-per-task="
    str2=""
    line=$(grep -n "$str1.*$str2" slurm_neuron_mpi_openmp.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_mpi_openmp.sb

    # str1="mpicc -fopenmp run_mpi_openmp.c neuron.c -o run_mpi_openmp_4_"
    # str2=" -lm"
    # line=$(grep -n "$str1.*$str2" slurm_neuron_mpi_openmp.sb | cut -d ":" -f 1)
    # sed -i "${line}s|$str1.*$str2|$str1$num$str2|" slurm_neuron_mpi_openmp.sb

    str1="    srun --mpi=pmi2 --ntasks=${num_process} --cpus-per-task="
    str2=""
    line=$(grep -n "$str1.*$str2" slurm_neuron_mpi_openmp.sb | cut -d ":" -f 1)
    sed -i "${line}s|$str1.*$str2|$str1$num ./run_mpi_openmp $num$str2|" slurm_neuron_mpi_openmp.sb

    # str1="# define N_THREADS_PER_RANK "
    # str2=""
    # line=$(grep -n "$str1.*$str2" run_mpi_openmp.c | cut -d ":" -f 1)
    # sed -i "${line}s|$str1.*$str2|$str1$num$str2|" run_mpi_openmp.c

    sbatch slurm_neuron_mpi_openmp.sb
done