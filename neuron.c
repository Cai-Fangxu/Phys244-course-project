# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include "neuron.h"

void neuron_euler(double* new_state, double* state, double i_ext, double dt){
/* 
    new_state or state = [V, m, h, n]. Hodgkin-Huxley neuron has 4 vairables.
    V is the voltage; m, h, n are gate varibales of ion channels. 
    i_ext: external current (unit: nA), including synaptic currents and driving currents(current from experimental device).
    dt: time spacing (unit: ms)
*/

    double V, m, h, n;
    double dVdt, dmdt, dhdt, dndt;

    double cap = 1.0; // neuron membrane capacitance (unit: nF)
    double gn = 120.0, gk = 20.0, gl = 0.3; // maximum conductance for Na, K, and leak current. (unit: muS)
    double vna = 50.0, vk = -77.0, vl = -54.4; // reversal potential for Na, K, and leak current. (unit: mV)
    
    // parameters used in the dynamics of subunits of ion channels
    double vm = -40.0, vn = -55.0, vh = -60.0;
    double dvm = 15.0, dvn = 30.0, dvh = -15.0;
    double tm0 = 0.1, tn0 = 1.0, th0 = 1.0;
    double tm1 = 0.4, tn1 = 5.0, th1 = 7.0;

    V = state[0], m = state[1], h = state[2], n = state[3];
    // currents with unit nA
    double i_Na = gn*pow(m, 3)*h*(vna - V); // Na current
    double i_K = gk*pow(n, 4)*(vk - V); // K current
    double i_L = gl*(vl - V); // leak current

    dVdt = (i_Na + i_K + i_L + i_ext)/cap;
    dmdt = (0.5 + 0.5*tanh((V - vm)/dvm) - m)/(tm0 + tm1 - tm1*pow(tanh((V-vm)/dvm), 2));
    dhdt = (0.5 + 0.5*tanh((V - vh)/dvh) - h)/(th0 + th1 - th1*pow(tanh((V-vh)/dvh), 2));
    dndt = (0.5 + 0.5*tanh((V - vn)/dvn) - n)/(tn0 + tn1 - tn1*pow(tanh((V-vn)/dvn), 2));

    new_state[0] = V + dVdt*dt;
    new_state[1] = m + dmdt*dt;
    new_state[2] = h + dhdt*dt;
    new_state[3] = n + dndt*dt;
}
