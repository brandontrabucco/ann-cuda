#ifndef NETWORKKERNELS_H_
#define NETWORKKERNELS_H_

#include <cuda.h>
#include "Neuron.cuh"
#include "PassiveNeuron.cuh"
#include "Synapse.cuh"
#include <vector>

using namespace std;

__global__ void activateInputNeuron(double *input, PassiveNeuron nodes[], double scalar, double *output);
__global__ void activateNeuron(double *input, Neuron nodes[], double *output);
__global__ void activateSynapse(double *input, Synapse connections[], double *output);
__global__ void sumInputFromSynapse(double *input, double *output, int nConnections);
__global__ void gradientDescent(double *error, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[]);
__global__ void sumWeightedError(double *error, Neuron nodes[], Synapse connections[], double *output, int nConnections);
__global__ void testK();
vector<int> factor(int f);

#endif
