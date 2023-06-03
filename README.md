# Phys244 Course Project
Course project for Phys 244 Parallel Computing for Science and Engineering.  
This project utilizes parallel computing to simulate a network of biophysical neurons.  

Biophysical properties of a neuron are defined in file neuron.c.  
The data to be read/written by the program is pre/post-processed by the jupyter notebook. Run the notebook to generate weight matrix of the network, initial conditions, and externally applied driving currents.  

After running the jupyter notebook, to compile and run the code parallelized by OpenMP, run: 
``` bash
gcc -fopenmp run_openmp.c neuron.c -o run_openmp -lm
./run_openmp
```

To compile and run the code parallelized by MPI, run:
```bash
mpicc run_mpi.c neuron.c -o run_mpi -lm
mpirun -np <num_processes> ./run_mpi
```

To compile and run the code parallelized by MPI + OpenMp, run:
```bash
mpicc -fopenmp run_mpi_openmp.c neuron.c -o run_mpi_openmp -lm
mpirun -np <num_processes> ./run_mpi_openmp
```

The voltage data of recorded neurons can be read and plot by the jupyter notebook. 
