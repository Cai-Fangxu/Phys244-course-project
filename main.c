# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include "neuron.h"

# define N_NEURON 64
# define N_THREADS 8

void initialize_array_from_file(double *array_ptr, int n_elemets, char* filename);
void write_array_to_file(double *array_ptr, int n_elemets, char* filename);

void initialize_array_from_file(double *array_ptr, int n_elemets, char* filename){
    int i; 
    FILE *file;
    int chunk_size = n_elemets / N_THREADS;
    # pragma omp parallel private(i, file)
    {
       file = fopen(filename, "rb");
       # pragma omp for
        for (i=0; i<N_THREADS; i++){
            fseek(file, chunk_size*i*sizeof(double), SEEK_SET);
            if (i == N_THREADS - 1){
                fread(&array_ptr[i*chunk_size], sizeof(double), n_elemets - (N_THREADS-1)*chunk_size, file);
            }
            else{
                fread(&array_ptr[i*chunk_size], sizeof(double), chunk_size, file);
            }
        }
        fclose(file);
    }
}

void write_array_to_file(double *array_ptr, int n_elemets, char* filename){
    int i;
    FILE *file; 
    int chunk_size = n_elemets / N_THREADS;
    # pragma omp parallel private(i, file)
    {
        file = fopen(filename, "wb");
        # pragma omp for
        for (i=0; i<N_THREADS; i++){
            fseek(file, chunk_size*i*sizeof(double), SEEK_SET);
            if (i == N_THREADS - 1){
                fwrite(&array_ptr[i*chunk_size], sizeof(double), n_elemets - (N_THREADS-1)*chunk_size, file);
            }
            else{
                fwrite(&array_ptr[i*chunk_size], sizeof(double), chunk_size, file);
            }
        }
        fclose(file);
    }
}

int main(){
    double w[N_NEURON][N_NEURON]; // weight matrix, w[i][j] is the weight from i to j
    double w_max = 1.0; // the range of w will be [-w_max/sqrt(N_NEURON), w_max/sqrt(N_NEURON)]
    int n_steps = 10000; //number of time steps
    double dt = 0.02; // time step used in the Euler method, the unit is ms
    double states[N_NEURON][4]; // each state of the neuron is ordered as [V, m, h, n]
    double ext_Is[N_NEURON] = {0.}; // external current that goes into each neuron (this is a sum of synaptic currents and driving currents (current from experimental devices)), the unit is nA. 

    int recorded_neuron_idx_list[] = {0, 1, 3, 5, 6, 9, 10, 11}; // a list of neurons whose voltages are to be recorded. 
    int record_size = sizeof(recorded_neuron_idx_list) / sizeof(recorded_neuron_idx_list[0]); // the number of neurons to be recorded. 
    double voltage_record[record_size][n_steps];

    int driven_neuron_idx_list[] = {0, 1}; // a list of neurons that are driven by driving currents. 
    int n_driven_neurons = sizeof(driven_neuron_idx_list) / sizeof(driven_neuron_idx_list[0]);
    double driving_currents[n_driven_neurons][n_steps];

    FILE *file;
    int main_seed = 123; 
    int seed; // private seed used in each thread

    int i, j;

    omp_set_num_threads(N_THREADS);

    /*
        Initialize weight matrix, neuronal states, and driving current, .
    */
    double *ptr_w = &w[0][0];
    initialize_array_from_file(ptr_w, N_NEURON*N_NEURON, "weights.bin");
    double *ptr_states = &states[0][0];
    initialize_array_from_file(ptr_states, N_NEURON*4, "initial_states.bin");
    double *ptr_driving_currents = &driving_currents[0][0];
    initialize_array_from_file(ptr_driving_currents, n_driven_neurons*n_steps, "driving_currents.bin");

    /*
    Run the simulation
    */
    # pragma omp parallel private(i, j)
    {
        for (int t=0; t<n_steps; t++){
            //one time step
            # pragma omp for // move one time step forward
            for (i=0; i<N_NEURON; i++){
                neuron_euler(states[i], states[i], ext_Is[i], dt);
            }
            # pragma omp barrier

            # pragma omp for // sum the synaptic currents
            for (i=0; i<N_NEURON; i++){
                ext_Is[i] = 0.;
                for (j=0; j<N_NEURON; j++){
                    ext_Is[i] += log(1 + exp(w[j][i]*(states[j][0] - states[i][0])));
                }
            }
            # pragma omp barrier
            # pragma omp for // add the driving current to ext_Is
            for (i=0; i<n_driven_neurons; i++){
                ext_Is[driven_neuron_idx_list[i]] += driving_currents[i][t];
            }
            # pragma omp for // record selected voltage data
            for (i=0; i<record_size; i++){
                voltage_record[i][t] = states[recorded_neuron_idx_list[i]][0];
            }
            # pragma omp barrier

        }
    }


    /*
    Write data to file
    */
    double *ptr_voltage_record = &voltage_record[0][0];
    write_array_to_file(ptr_voltage_record, record_size*n_steps, "voltage_record.bin");
    
    return 0;
}