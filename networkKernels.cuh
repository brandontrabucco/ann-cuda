#ifndef NETWORKKERNELS_H_
#define NETWORKKERNELS_H_

#include <cuda.h>
#include "Neuron.cuh"
#include "PassiveNeuron.cuh"
#include "Synapse.cuh"
#include <vector>

using namespace std;
vector<int> factor(int f);

/**
 *
 * 	Thread managed CUDA
 *
 */
__global__ void inputNeuronKernel(double *input, PassiveNeuron nodes[], double scalar, double *output, int nNeurons, int nPerThread, int nThreads);
__global__ void neuronKernel(double *input, Neuron nodes[], double *output, int nNeurons, int nPerThread, int nThreads);
__global__ void synapseKernel(double *input, Synapse connections[], double *output, int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nPerBlock, int nThreads, int nBlocks);
__global__ void sumInputKernel(double *input, double *output, int nConnectionsPer, int nNeurons, int nPerThread, int nThreads);
__global__ void hiddenLayerGradientDescentKernel(double *errorPrime, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nPerBlock, int nThreads, int nBlocks);
__global__ void outputLayerGradientDescentKernel(double *errorPrime, double learningRate, Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious, int nPerThread, int nPerBlock, int nThreads, int nBlocks);
__global__ void hiddenLayerSumErrorKernel(double *errorPrime, Neuron nodes[], Synapse connections[], double *output, int nConnectionsPer, int nNeurons, int nPerThread, int nThreads);
__global__ void outputLayerSumErrorKernel(double *errorPrime, Synapse connections[], double *output, int nConnectionsPer, int nNeurons, int nPerThread, int nThreads);

class KernelAdapter {
private:
	static const int nThreads = 1024;
	static const int nBlocks = 1024;
	static int nPerThread;
	static int nPerBlock;
public:
	static void startInputNeuronKernel(double *input, PassiveNeuron nodes[], double scalar, double *output, int nNeurons);
	static void startNeuronKernel(double *input, Neuron nodes[], double *output, int nNeurons);
	static void startSynapseKernel(double *input, Synapse connections[], double *output, int nNeuronsCurrent, int nNeuronsPrevious);
	static void startSumInputKernel(double *input, double *output, int nConnectionsPer, int nNeurons);
	static void startHiddenLayerGradientDescentKernel(double *errorPrime, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious);
	static void startOutputLayerGradientDescentKernel(double *errorPrime, double learningRate, Neuron previous[], Synapse connections[], int nNeuronsCurrent, int nNeuronsPrevious);
	static void startHiddenLayerSumErrorKernel(double *errorPrime, Neuron nodes[], Synapse connections[], double *output, int nConnectionsPer, int nNeurons);
	static void startOutputLayerSumErrorKernel(double *errorPrime, Synapse connections[], double *output, int nConnectionsPer, int nNeurons);
};

#endif
