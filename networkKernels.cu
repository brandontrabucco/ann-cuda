#include "networkKernels.cuh"

__global__ void activateInputNeuron(double *input, PassiveNeuron nodes[], double *scalar, double *output) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	output[idx] = nodes[idx].get(input[idx], scalar[0]);
}

__global__ void activateNeuron(double *input, Neuron nodes[], double *output) {
	int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;	// the current layer
	output[blockId] = nodes[blockId].get(input[blockId]);
}

__global__ void activateSynapse(double *input, Synapse connections[], double *output) {
	int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;	// the current layer
	int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the previous layer
	output[blockId * (blockDim.x * blockDim.y) + threadId] = connections[blockId * (blockDim.x * blockDim.y) + threadId].get(input[threadId]);
}

__global__ void sumInputFromSynapse(double *input, double *output) {
	int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;	// the current layer
	int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the previous layer
	output[blockId] += input[blockId * (blockDim.x * blockDim.y) + threadId];
}

__global__ void gradientDescent(double *error, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[]) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;	// the current layer
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;	// the previous layer
	connections[(idx * (gridDim.y * blockDim.y)) + idy].weight -= learningRate * error[idx] * nodes[idx].derivative * previous[idy].activation;
	//connections[(idx * (gridDim.y * blockDim.y)) + idy].bias -= learningRate * error[idx] * nodes[idx].derivative;
}

__global__ void sumWeightedError(double *error, Neuron nodes[], Synapse connections[], double *output) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;	// the current layer
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;	// the previous layer
	output[idy] += ((error[idx] * nodes[idx].derivative) * connections[(idx * (gridDim.y * blockDim.y)) + idy].weight) + connections[(idx * (gridDim.y * blockDim.y)) + idy].bias;
}

__global__ void testK() {
	printf("This is a test message!!\n");
}
