# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include "neuron.h"

# define N_NEURON 1024

void initialize_array_from_file(double *array_ptr, int n_elemets, char* filename, int n_threads);
void write_array_to_file(double *array_ptr, int n_elemets, char* filename, int n_threads);
double **alloc_2d_double(int rows, int columns); // allocate memory for a 2d double array so that they are contiguous. 

void initialize_array_from_file(double *array_ptr, int n_elemets, char* filename, int n_threads){
    int i; 
    FILE *file;
    int chunk_size = n_elemets / n_threads;
    # pragma omp parallel private(i, file)
    {
       file = fopen(filename, "rb");
       # pragma omp for
        for (i=0; i<n_threads; i++){
            fseek(file, chunk_size*i*sizeof(double), SEEK_SET);
            if (i == n_threads - 1){
                fread(&array_ptr[i*chunk_size], sizeof(double), n_elemets - (n_threads-1)*chunk_size, file);
            }
            else{
                fread(&array_ptr[i*chunk_size], sizeof(double), chunk_size, file);
            }
        }
        fclose(file);
    }
}

void write_array_to_file(double *array_ptr, int n_elemets, char* filename, int n_threads){
    int i;
    FILE *file; 
    int chunk_size = n_elemets / n_threads;
    # pragma omp parallel private(i, file)
    {
        file = fopen(filename, "wb");
        # pragma omp for
        for (i=0; i<n_threads; i++){
            fseek(file, chunk_size*i*sizeof(double), SEEK_SET);
            if (i == n_threads - 1){
                fwrite(&array_ptr[i*chunk_size], sizeof(double), n_elemets - (n_threads-1)*chunk_size, file);
            }
            else{
                fwrite(&array_ptr[i*chunk_size], sizeof(double), chunk_size, file);
            }
        }
        fclose(file);
    }
}

double **alloc_2d_double(int rows, int columns){
    double *data = (double *)malloc(rows*columns*sizeof(double));
    double **array = (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++){
        array[i] = &(data[columns*i]);
    }
    return array;
}

int main(int argc, char *argv[]){
    if (argc < 2) {
        printf("Please provide the number of threads when running the program\n");
        return 1;
    }
    int n_threads = atoi(argv[1]);

    double start_time, end_time;

    double **w; // weight matrix, w[i][j] is the weight from j to i, not i to j. 
    int n_steps = 20000; //number of time steps
    double dt = 0.02; // time step used in the Euler method, the unit is ms
    double states[N_NEURON][4]; // each state of the neuron is ordered as [V, m, h, n]
    double ext_Is[N_NEURON] = {0.}; // external current that goes into each neuron (this is a sum of synaptic currents and driving currents (current from experimental devices)), the unit is nA. 

    int recorded_neuron_idx_list[] = {0, 46, 67, 69, 70, 72, 108, 117, 118, 135, 139, 147, 159, 178, 179, 202, 208, 215, 218, 219, 241, 243, 255, 295, 315, 342, 386, 396, 435, 445, 465, 472, 474, 508, 515, 553, 572, 574, 578, 614, 633, 653, 658, 659, 666, 668, 679, 702, 715, 736, 744, 763, 846, 852, 870, 892, 893, 923, 942, 966, 981, 982, 984, 1011}; // a list of neurons whose voltages are to be recorded. 
    int record_size = sizeof(recorded_neuron_idx_list) / sizeof(recorded_neuron_idx_list[0]); // the number of neurons to be recorded. 
    double **voltage_record;

    int driven_neuron_idx_list[] = {1, 50, 79, 94, 112, 164, 238, 247, 313, 344, 371, 406, 452, 481, 554, 557, 565, 640, 668, 681, 695, 725, 761, 778, 839, 866, 896, 937, 971, 986, 1006, 1012}; // a list of neurons that are driven by driving currents. 
    int n_driven_neurons = sizeof(driven_neuron_idx_list) / sizeof(driven_neuron_idx_list[0]);
    double **driving_currents;

    FILE *file;
    int i, j;

    start_time = omp_get_wtime();

    omp_set_num_threads(n_threads);

    /*
        Initialize weight matrix, neuronal states, and driving current, .
    */
    /* allocate memory for w, voltage_record and driving_currents */
    w = alloc_2d_double(N_NEURON, N_NEURON);
    voltage_record = alloc_2d_double(record_size, n_steps);
    driving_currents = alloc_2d_double(n_driven_neurons, n_steps);

    double *ptr_w = &w[0][0];
    // initialize_array_from_file(ptr_w, N_NEURON*N_NEURON, "/expanse/lustre/scratch/fcai/temp_project/1024_16_32/weights_1024.bin", n_threads);
    double *ptr_states = &states[0][0];
    // initialize_array_from_file(ptr_states, N_NEURON*4, "/expanse/lustre/scratch/fcai/temp_project/1024_16_32/initial_states_1024.bin", n_threads);
    double *ptr_driving_currents = &driving_currents[0][0];
    // initialize_array_from_file(ptr_driving_currents, n_driven_neurons*n_steps, "/expanse/lustre/scratch/fcai/temp_project/1024_16_32/driving_currents_1024_16.bin", n_threads);
    file = fopen("/expanse/lustre/scratch/fcai/temp_project/1024_32_64/weights_1024.bin", "rb");
    fread(ptr_w, sizeof(double), N_NEURON*N_NEURON, file);
    fclose(file);
    file = fopen("/expanse/lustre/scratch/fcai/temp_project/1024_32_64/initial_states_1024.bin", "rb");
    fread(ptr_states, sizeof(double), N_NEURON*4, file);
    fclose(file);
    file = fopen("/expanse/lustre/scratch/fcai/temp_project/1024_32_64/driving_currents_1024_32.bin", "rb");
    fread(ptr_driving_currents, sizeof(double), n_driven_neurons*n_steps, file);
    fclose(file);


    /*
    Run the simulation
    */
    # pragma omp parallel private(i, j)
    {
        for (int t=0; t<n_steps; t++){
            # pragma omp for // use Euler method to move forward one time step.
            for (i=0; i<N_NEURON; i++){
                neuron_euler(states[i], states[i], ext_Is[i], dt);
            }
            # pragma omp barrier

            # pragma omp for // sum the synaptic currents
            for (i=0; i<N_NEURON; i++){
                ext_Is[i] = 0.;
                for (j=0; j<N_NEURON; j++){
                    // ext_Is[i] += log(1 + exp(w[i][j]*(states[j][0] - states[i][0])));
                    ext_Is[i] += w[i][j]/(1 + exp(-(states[j][0] - states[i][0])/20.));
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
    write_array_to_file(ptr_voltage_record, record_size*n_steps, "/expanse/lustre/scratch/fcai/temp_project/1024_32_64/voltage_record_openmp.bin", n_threads);

    end_time = omp_get_wtime(); 
    printf("Execution Time: %f seconds\n", end_time-start_time);
    
    free(w[0]); free(voltage_record[0]); free(driving_currents[0]);
    free(w); free(voltage_record); free(driving_currents); 
    return 0;
}