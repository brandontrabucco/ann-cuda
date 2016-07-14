#include "networkKernels.cuh"

__global__ void inputNeuronKernel(double *input, PassiveNeuron nodes[], double scalar, double *output, int nNeurons, int nPerThread, int nThreads) {
	for (int i = 0; i < nPerThread; i++) {
		int neuronId = threadIdx.x + (i * nThreads);
		if (neuronId < nNeurons) {
			output[neuronId] = nodes[neuronId].get(input[neuronId], scalar);
		}
	}
}

__global__ void neuronKernel(double *input, Neuron nodes[], double *output, int nNeurons, int nPerThread, int nThreads) {
	for (int i = 0; i < nPerThread; i++) {
		int neuronId = threadIdx.x + (i * nThreads);
		if (neuronId < nNeurons) {
			output[neuronId] = nodes[neuronId].get(input[neuronId]);
			//printf("output %f\n",output[neuronId]);
		}
	}
}

__global__ void synapseKernel(double *input, Synapse connections[], double *output, int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nPerBlock, int nThreads, int nBlocks) {
	for (int i = 0; i < nPerBlock; i++) {
		for (int j = 0; j < nPerThread; j++) {
			int blockId = blockIdx.x + (i * nBlocks);	// the current layer
			int threadId = threadIdx.x + (j * nThreads);	// the previous layer
			if (blockId < nNeuronsCurrent && threadId < nNeuronsPrevious) {
				output[blockId * (nNeuronsPrevious) + threadId] = connections[blockId * (nNeuronsPrevious) + threadId].get(input[threadId]);
				//printf("Input = %f : Output = %f\n", input[threadId], output[blockId * (nNeuronsPrevious) + threadId]);
			}
		}
	}
}

__global__ void sumInputKernel(double *input, double *output, int nConnectionsPer, int nNeurons, int nPerThread, int nThreads) {
	for (int i = 0; i < nPerThread; i++) {
		int neuronId = threadIdx.x + (i * nThreads);
		if (neuronId < nNeurons) for (int j = 0; j < nConnectionsPer; j++) {
			output[neuronId] +=	input[(neuronId * nConnectionsPer) + j];
			//printf("Sum %f\n", output[neuronId]);
		}
	}
}

__global__ void hiddenLayerGradientDescentKernel(double *errorPrime, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nPerBlock, int nThreads, int nBlocks) {
	for (int i = 0; i < nPerBlock; i++) {
		for (int j = 0; j < nPerThread; j++) {
			int blockId = blockIdx.x + (i * nBlocks);	// the current layer
			int threadId = threadIdx.x + (j * nThreads);	// the previous layer
			if (blockId < nNeuronsCurrent && threadId < nNeuronsPrevious) 	connections[blockId * (nNeuronsPrevious) + threadId].weight -= learningRate * errorPrime[blockId] * nodes[blockId].derivative * previous[threadId].activation;
		}
	}
}

__global__ void outputLayerGradientDescentKernel(double *errorPrime, double learningRate, Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nPerBlock, int nThreads, int nBlocks) {
	for (int i = 0; i < nPerBlock; i++) {
		for (int j = 0; j < nPerThread; j++) {
			int blockId = blockIdx.x + (i * nBlocks);	// the current layer
			int threadId = threadIdx.x + (j * nThreads);	// the previous layer
			if (blockId < nNeuronsCurrent && threadId < nNeuronsPrevious) {
				connections[blockId * (nNeuronsPrevious) + threadId].weight -= learningRate * errorPrime[blockId] * previous[threadId].activation;
			}
		}
	}
}

__global__ void hiddenLayerSumErrorKernel(double *errorPrime, Neuron nodes[], Synapse connections[], double *output, int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nThreads) {
	for (int i = 0; i < nPerThread; i++) {
		int neuronId = threadIdx.x + (i * nThreads);
		if (neuronId < nNeuronsPrevious) for (int j = 0; j < (nNeuronsCurrent); j++) {
			output[neuronId] += ((errorPrime[j] * nodes[neuronId].derivative) * connections[(j * nNeuronsPrevious) + neuronId].weight);
		}
	}
}

__global__ void outputLayerSumErrorKernel(double *errorPrime, Synapse connections[], double *output, int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nThreads) {
	for (int i = 0; i < nPerThread; i++) {
		int neuronId = threadIdx.x + (i * nThreads);
		if (neuronId < nNeuronsPrevious) for (int j = 0; j < (nNeuronsCurrent); j++) {
			output[neuronId] += (errorPrime[j] * connections[(j * nNeuronsPrevious) + neuronId].weight);
		}
	}
}

__global__ void testK(double *errorPrime, double learningRate, Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nPerBlock, int nThreads, int nBlocks) {
	printf("Test print from a kernel!!\n");
}

int KernelAdapter::nPerThread = 0;
int KernelAdapter::nPerBlock = 0;

void KernelAdapter::startInputNeuronKernel(double *input, PassiveNeuron nodes[], double scalar, double *output, int nNeurons) {
	KernelAdapter::nPerThread = ceil(((double)nNeurons) / ((double)KernelAdapter::nThreads));
	inputNeuronKernel<<<1, KernelAdapter::nThreads>>>(input, nodes, scalar, output, nNeurons, nPerThread, nThreads);
	cudaDeviceSynchronize();
}

void KernelAdapter::startNeuronKernel(double *input, Neuron nodes[], double *output, int nNeurons) {
	KernelAdapter::nPerThread = ceil(((double)nNeurons) / ((double)KernelAdapter::nThreads));
	neuronKernel<<<1, KernelAdapter::nThreads>>>(input, nodes, output, nNeurons, KernelAdapter::nPerThread, KernelAdapter::nThreads);
	cudaDeviceSynchronize();
}

void KernelAdapter::startSynapseKernel(double *input, Synapse connections[], double *output, int nNeuronsCurrent, int nNeuronsPrevious) {
	KernelAdapter::nPerThread = ceil(((double)nNeuronsPrevious) / ((double)KernelAdapter::nThreads));
	KernelAdapter::nPerBlock = ceil(((double)nNeuronsCurrent) / ((double)KernelAdapter::nBlocks));
	synapseKernel<<<KernelAdapter::nBlocks, KernelAdapter::nThreads>>>(input, connections, output, nNeuronsCurrent, nNeuronsPrevious, KernelAdapter::nPerThread, KernelAdapter::nPerBlock, KernelAdapter::nThreads, KernelAdapter::nBlocks);
	cudaDeviceSynchronize();
}

void KernelAdapter::startSumInputKernel(double *input, double *output, int nNeuronsCurrent, int nNeuronsPrevious) {
	KernelAdapter::nPerThread = ceil(((double)nNeuronsCurrent) / ((double)KernelAdapter::nThreads));
	sumInputKernel<<<1, KernelAdapter::nThreads>>>(input, output, nNeuronsPrevious, nNeuronsCurrent, KernelAdapter::nPerThread, KernelAdapter::nThreads);
	cudaDeviceSynchronize();
}

void KernelAdapter::startHiddenLayerGradientDescentKernel(double *errorPrime, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious) {
	KernelAdapter::nPerThread = ceil(((double)nNeuronsPrevious) / ((double)KernelAdapter::nThreads));
	KernelAdapter::nPerBlock = ceil(((double)nNeuronsCurrent) / ((double)KernelAdapter::nBlocks));
	hiddenLayerGradientDescentKernel<<<KernelAdapter::nBlocks, KernelAdapter::nThreads>>>(errorPrime, learningRate, nodes, previous, connections, nNeuronsCurrent, nNeuronsPrevious, KernelAdapter::nPerThread, KernelAdapter::nPerBlock, KernelAdapter::nThreads, KernelAdapter::nBlocks);
	cudaDeviceSynchronize();
}

void KernelAdapter::startOutputLayerGradientDescentKernel(double *errorPrime, double learningRate, Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious) {
	KernelAdapter::nPerThread = ceil(((double)nNeuronsPrevious) / ((double)KernelAdapter::nThreads));
	KernelAdapter::nPerBlock = ceil(((double)nNeuronsCurrent) / ((double)KernelAdapter::nBlocks));
	outputLayerGradientDescentKernel<<<KernelAdapter::nBlocks, KernelAdapter::nThreads>>>(errorPrime, learningRate, previous, connections, nNeuronsCurrent, nNeuronsPrevious, KernelAdapter::nPerThread, KernelAdapter::nPerBlock, KernelAdapter::nThreads, KernelAdapter::nBlocks);
	cudaDeviceSynchronize();
}

void KernelAdapter::startHiddenLayerSumErrorKernel(double *errorPrime, Neuron nodes[], Synapse connections[], double *output, int nNeuronsCurrent, int nNeuronsPrevious) {
	KernelAdapter::nPerThread = ceil(((double)nNeuronsPrevious) / ((double)KernelAdapter::nThreads));
	hiddenLayerSumErrorKernel<<<1, KernelAdapter::nThreads>>>(errorPrime, nodes, connections, output, nNeuronsCurrent, nNeuronsPrevious, KernelAdapter::nPerThread, KernelAdapter::nThreads);
	cudaDeviceSynchronize();
}

void KernelAdapter::startOutputLayerSumErrorKernel(double *errorPrime, Synapse connections[], double *output, int nNeuronsCurrent, int nNeuronsPrevious) {
	KernelAdapter::nPerThread = ceil(((double)nNeuronsPrevious) / ((double)KernelAdapter::nThreads));
	outputLayerSumErrorKernel<<<1, KernelAdapter::nThreads>>>(errorPrime, connections, output, nNeuronsCurrent, nNeuronsPrevious, KernelAdapter::nPerThread, KernelAdapter::nThreads);
	cudaDeviceSynchronize();
}
