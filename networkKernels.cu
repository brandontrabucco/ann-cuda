#include "networkKernels.cuh"

__global__ void activateInputNeuron(double *input, PassiveNeuron nodes[], double *scalar, double *output) {
	int neuronId = (threadIdx.y * blockDim.x) + threadIdx.x;
	output[neuronId] = nodes[neuronId].get(input[neuronId], scalar[0]);
}

__global__ void activateNeuron(double *input, Neuron nodes[], double *output) {
	int neuronId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the current layer
	output[neuronId] = nodes[neuronId].get(input[neuronId]);
	//printf("Neurons activating %f\n", output[neuronId]);
}

__global__ void activateSynapse(double *input, Synapse connections[], double *output) {
	int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;	// the current layer
	int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the previous layer
	output[blockId * (blockDim.x * blockDim.y) + threadId] = connections[blockId * (blockDim.x * blockDim.y) + threadId].get(input[threadId]);
	//printf("Input = %f\n", output[blockId * (blockDim.x * blockDim.y) + threadId]);
}

__global__ void sumInputFromSynapse(double *input, double *output, int nConnectionsPerNeuron) {
	// need to make summation parallel but atomic
	int neuronId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the current layer
	for (int i = 0; i < nConnectionsPerNeuron; i++) {
		output[neuronId] +=	input[(neuronId * nConnectionsPerNeuron) + i];
	}

	//printf("Sum = %f\n", output[neuronId]);
}

__global__ void gradientDescent(double *error, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[]) {
	int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;	// the current layer
	int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the previous layer
	connections[blockId * (blockDim.x * blockDim.y) + threadId].weight -= learningRate * error[blockId] * nodes[blockId].derivative * previous[threadId].activation;
	//connections[blockId * (blockDim.x * blockDim.y) + threadId].bias -= learningRate * error[blockId] * nodes[blockId].derivative;
}

__global__ void sumWeightedError(double *error, Neuron nodes[], Synapse connections[], double *output, int nConnectionsPreNeuron) {
	int neuronId = (threadIdx.y * blockDim.x) + threadIdx.x;

	for (int i = 0; i < nConnectionsPreNeuron; i++) {
		output[neuronId] += ((error[i] * nodes[i].derivative) * connections[(i * nConnectionsPreNeuron) + neuronId].weight) + connections[(i * nConnectionsPreNeuron) + neuronId].bias;
	}

	//printf("Error sum %f", output[neuronId]);
}

__global__ void testK() {
	printf("This is a test message!!\n");
}

vector<int> factor(int f)
{
	vector<int> factors = {1, 1};
	bool s = false;
    for(int ii = 2; ii<=f; ii++) {
        while(f % ii == 0) {
            f = f/ii;
            if (s) {
            	factors[0] *= ii ;
            	s = !s;
            } else {
            	factors[1] *= ii ;
            	s = !s;
            }
        }
    }

    return factors;
}
