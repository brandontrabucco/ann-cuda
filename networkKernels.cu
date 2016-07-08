#include "networkKernels.cuh"

__global__ void activateInputNeuron(double *input, PassiveNeuron nodes[], double scalar, double *output) {
	int neuronId = (threadIdx.y * blockDim.x) + threadIdx.x;
	output[neuronId] = nodes[neuronId].get(input[neuronId], scalar);
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
	//printf("Input = %f : Output = %f\n", input[threadId], output[blockId * (blockDim.x * blockDim.y) + threadId]);
}

__global__ void sumInputFromSynapse(double *input, double *output, int nConnections) {
	int neuronId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the current layer
	for (int i = 0; i < nConnections; i++) {
		output[neuronId] +=	input[(neuronId * nConnections) + i];
		//if (neuronId == 0)printf("From %f\n", input[(neuronId * nConnections) + i]);
	}

	//if (neuronId == 0)printf("Sum = %f\n", output[neuronId]);
}

__global__ void gradientDescentHiddenLayer(double *errorPrime, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[]) {
	int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;	// the current layer
	int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the previous layer

	//if (blockId == 6 && threadId == 0) printf("Delta %f\n", learningRate * error[blockId] * nodes[blockId].derivative * previous[threadId].activation);
	//if (blockId == 6 && threadId == 0) printf("Learning %f\n", learningRate);
	//if (blockId == 6 && threadId == 0) printf("Error %f\n", errorPrime[blockId]);
	//if (blockId == 6 && threadId == 0) printf("Derivative %f\n", nodes[blockId].derivative);
	//if (blockId == 6 && threadId == 0) printf("Activation %f\n\n", previous[threadId].activation);



	connections[blockId * (blockDim.x * blockDim.y) + threadId].weight -= learningRate * errorPrime[blockId] * nodes[blockId].derivative * previous[threadId].activation;
	//connections[blockId * (blockDim.x * blockDim.y) + threadId].bias -= learningRate * errorPrime[blockId] * nodes[blockId].derivative;

	//if (blockId == 0 && threadId == 0) printf("Weight %f\n", connections[blockId * (blockDim.x * blockDim.y) + threadId].weight);
}

__global__ void gradientDescentOutputLayer(double *errorPrime, double learningRate, Neuron previous[], Synapse connections[]) {
	int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;	// the current layer
	int threadId = (threadIdx.y * blockDim.x) + threadIdx.x;	// the previous layer

	//if (blockId == 6 && threadId == 0) printf("Delta %f\n", learningRate * error[blockId] * nodes[blockId].derivative * previous[threadId].activation);
	//if (blockId == 6 && threadId == 0) printf("Learning %f\n", learningRate);
	//if (blockId == 6 && threadId == 0) printf("Error %f\n", error[blockId]);
	//if (blockId == 6 && threadId == 0) printf("Derivative %f\n", nodes[blockId].derivative);
	//if (blockId == 6 && threadId == 0) printf("Activation %f\n\n", previous[threadId].activation);



	connections[blockId * (blockDim.x * blockDim.y) + threadId].weight -= learningRate * errorPrime[blockId] * previous[threadId].activation;
	//connections[blockId * (blockDim.x * blockDim.y) + threadId].bias -= learningRate * errorPrime[blockId];

	//if (blockId == 0 && threadId == 0) printf("Weight %f\n", connections[blockId * (blockDim.x * blockDim.y) + threadId].weight);
}

__global__ void sumWeightedError(double *error, Neuron nodes[], Synapse connections[], double *output, int nConnections) {
	int neuronId = (threadIdx.y * blockDim.x) + threadIdx.x;
	for (int i = 0; i < (nConnections); i++) {
		output[neuronId] += ((error[i] * nodes[i].derivative) * connections[(i * nConnections) + neuronId].weight);	// error is in this line
	}
	//printf("Sum %f at %d\n", output[neuronId], neuronId);
}

__global__ void testK() {
	printf("This is a test message!!\n");
}

vector<int> factor(int f) {
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
    } return factors;
}
