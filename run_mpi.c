# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <mpi.h>
# include "neuron.h"

# define N_NEURON 64

double **alloc_2d_double(int rows, int columns); // allocate memory for a 2d double array so that they are contiguous. 

double **alloc_2d_double(int rows, int columns){
    double *data = (double *)malloc(rows*columns*sizeof(double));
    double **array = (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++){
        array[i] = &(data[columns*i]);
    }
    return array;
}

int main(int argc, char *argv[]){
    int rank, n_process;
    int n_steps = 10000; //number of time steps
    double dt = 0.02; // time step used in the Euler method, the unit is ms
    int n_neuron_local; // number of neurons stored in this process
    double voltages[N_NEURON];
    double **w; // weight matrix, w[i][j] is the weight from j to i, not i to j. The size of w should be (n_neuron_local, N_NEURON). 
    double **states; // a 2d array, each state of the neuron is ordered as [V, m, h, n]. The size of states is (n_neuron_local, 4).
    double *ext_Is; // external current that goes into each neuron (this is a sum of synaptic currents and driving currents (current from experimental devices)), the unit is nA. The size should be (n_neuron_local, )

    int recorded_neuron_idx_list[] = {0, 1, 3, 5, 6, 9, 10, 30}; // a list of neurons whose voltages are to be recorded. 
    int record_size = sizeof(recorded_neuron_idx_list) / sizeof(recorded_neuron_idx_list[0]);
    int record_size_local; // the number of neurons to be recorded within this rank. Note that since the voltage of all neurons will be available in each rank. The recorded neuron is not necessarily the neuron simulated in this rank. 
    int *recorded_neuron_idx_list_local; // a list of indicies that correspond to the recorded neurons. These indicies are global indicies, from 0 to N_NEURON-1. 
    double **voltage_record; // The size should be (record_size_local, n_steps)

    int driven_neuron_idx_list[] = {0, 1}; // a list of neurons that are driven by driving currents. 
    int n_driven_neuron = sizeof(driven_neuron_idx_list) / sizeof(driven_neuron_idx_list[0]);
    int n_driven_neruon_local; // the number of neurons that are have a driving current within the rank. 
    int *driven_neuron_idx_list_local; // a list of indicies that correspond to the driven neurons. The indices are local indicies, from 0 to n_neuron_local-1. (local_index + first_local_neuron_idx = global_index)
    double **driving_currents; // the size is (n_driven_neuron_local, n_steps)

    int i, j, tmp_idx;
    int first_local_neuron_idx; // the indicies of neuron simulated in this rank: first_local_neuron_idx ~ first_local_neuron_idx + n_neuron_local
    int *recv_counts; // an array of number of neurons in each rank. Used in MPI_Allgatherv operation to gather voltages. 
    int *displs; //used in MPI_Allgatherv, store the displacement of receive buffer for each vector received. 
    FILE *file;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_process);
    MPI_Datatype column_type;

    n_neuron_local = N_NEURON / n_process;
    if (rank<N_NEURON%n_process) {n_neuron_local += 1;}

    if (rank<N_NEURON%n_process) {first_local_neuron_idx = rank*n_neuron_local;}
    else {first_local_neuron_idx = N_NEURON - (n_process-rank)*n_neuron_local;}

    record_size_local = record_size / n_process;
    if (rank<record_size%n_process) {record_size_local += 1;}

    n_driven_neruon_local = 0;
    for (i=0; i<n_driven_neuron; i++){
        if (driven_neuron_idx_list[i]>=first_local_neuron_idx && driven_neuron_idx_list[i]<first_local_neuron_idx+n_neuron_local){
            n_driven_neruon_local += 1;
        }
    }

/*------------------------------------dynamical memory allocation---------------------------------------------*/
    /* allocate memory for w */
    w = alloc_2d_double(n_neuron_local, N_NEURON);
    /* allocate memory for states */
    states = alloc_2d_double(n_neuron_local, 4);
    /*allocate memory for ext_Is*/
    ext_Is = (double *)malloc(n_neuron_local*sizeof(double));
    /* allocate memory for recorded_neuron_idx_list_local and voltage record*/
    if (record_size_local>0){
        recorded_neuron_idx_list_local = (int *)malloc(record_size_local*sizeof(int));
        voltage_record = alloc_2d_double(record_size_local, n_steps);
    }
    /*allocate memory for driven_neuron_idx_list_local and driving currents*/
    if (n_driven_neruon_local>0){
        driven_neuron_idx_list_local = (int *)malloc(n_driven_neruon_local*sizeof(int));
        driving_currents = alloc_2d_double(n_driven_neruon_local, n_steps);
    }
    /*allocate memory for recv_counts and displs*/
    recv_counts = (int *)malloc(n_process*sizeof(int));
    displs = (int *)malloc(n_process*sizeof(int));
/*-------------------------------------------------------------------------------------------------------*/

/*------------------------------------arrays initialization---------------------------------------------*/
    /*read w from file*/
    file = fopen("weights.bin", "rb");
    for (i=0; i<n_neuron_local; i++){
        fseek(file, sizeof(double)*(i+first_local_neuron_idx)*N_NEURON, SEEK_SET);
        fread(w[i], sizeof(double), N_NEURON, file);
    }
    fclose(file);
    /*read states from file*/
    file = fopen("initial_states.bin", "rb");
    for (i=0; i<n_neuron_local; i++){
        fseek(file, sizeof(double)*(i+first_local_neuron_idx)*4, SEEK_SET);
        fread(states[i], sizeof(double), 4, file);
    }
    fclose(file);
    /*initialize ext_Is to all 0s*/
    for (i=0; i<n_neuron_local; i++){
        ext_Is[i] = 0.;
    }
    /*fill recorded_neuron_idx_list_local*/
    if (rank<record_size%n_process) {tmp_idx = rank*record_size_local;}
    else {tmp_idx = record_size - (n_process-rank)*record_size_local;}
    for (i=0; i<record_size_local; i++){
        recorded_neuron_idx_list_local[i] = recorded_neuron_idx_list[tmp_idx+i];
    }
    /*fill driven_neuron_idx_list_local and driving_currents*/
    file = fopen("driving_currents.bin", "rb");
    for (tmp_idx=0, i=0; tmp_idx<n_driven_neuron; tmp_idx++){
        if (driven_neuron_idx_list[tmp_idx]>=first_local_neuron_idx && driven_neuron_idx_list[tmp_idx]<first_local_neuron_idx+n_neuron_local) {
            driven_neuron_idx_list_local[i] = driven_neuron_idx_list[tmp_idx] - first_local_neuron_idx;
            fseek(file, sizeof(double)*(tmp_idx)*n_steps, SEEK_SET);
            fread(driving_currents[i], sizeof(double), n_steps, file);
            i += 1;
        }
    }
    fclose(file);
    /*fill recv_counts and displs*/
    MPI_Allgather(&n_neuron_local, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (i=1; i<n_process; i++){
        displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
/*-------------------------------------------------------------------------------------------------------*/

    MPI_Type_vector(n_neuron_local, 1, 4, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
/*------------------------------------run simulation---------------------------------------------*/
    for (int t=0; t<n_steps; t++){
        for (i=0; i<n_neuron_local; i++){ // use Euler method to move forward one time step
            neuron_euler(states[i], states[i], ext_Is[i], dt);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgatherv(&states[0][0], 1, column_type, voltages, recv_counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        for (i=0; i<n_neuron_local; i++){ // sum the synaptic currents
            ext_Is[i] = 0.;
            for (j=0; j<N_NEURON; j++){
                ext_Is[i] += log(1 + exp(w[i][j]*(voltages[j] - voltages[i+first_local_neuron_idx])));
            }
        }

        for (i=0; i<n_driven_neruon_local; i++){ // add the driving current to ext_Is
            ext_Is[driven_neuron_idx_list_local[i]] += driving_currents[i][t];
        }

        for (i=0; i<record_size_local; i++){ // record selected voltage data
            voltage_record[i][t] = voltages[recorded_neuron_idx_list_local[i]];
        }
    }

    /*Write data to file*/
    if (rank<record_size%n_process) {tmp_idx = rank*record_size_local;}
    else {tmp_idx = record_size - (n_process-rank)*record_size_local;}
    file = fopen("voltage_record_mpi.bin", "wb");
    fseek(file, sizeof(double)*tmp_idx*n_steps, SEEK_SET);
    fwrite(&voltage_record[0][0], sizeof(double), record_size_local*n_steps, file);
    fclose(file);

    /*free dynamically allocated memory*/
    free(w[0]); free(states[0]);
    free(w); free(states); free(ext_Is); free(recv_counts); free(displs);
    if (record_size_local>0) {free(voltage_record[0]); free(voltage_record); free(recorded_neuron_idx_list_local);}
    if (n_driven_neruon_local>0) {free(driving_currents[0]); free(driving_currents); free(driven_neuron_idx_list_local);} 

    MPI_Finalize();
    return 0;
}