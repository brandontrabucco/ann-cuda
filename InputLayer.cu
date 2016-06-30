/*
 * InputLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "InputLayer.cuh"

__global__ void activateNeuron(double *input, PassiveNeuron nodes[], double scalar, double *output) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	output[idx] = ((PassiveNeuron)nodes[idx]).get(input[idx], scalar);
}

InputLayer::InputLayer(int w, double range, bool db) {
	// TODO Auto-generated constructor stub
	debug = db;
	currentLayerNeurons = w;
	scalar = 1 / range;
	for (int i = 0; i < currentLayerNeurons; i++) {
		PassiveNeuron n = PassiveNeuron();
		n.index = i;
		neurons.push_back(n);
		if (debug) cout << "Passive Neuron " << i << endl;
	}
}

InputLayer::~InputLayer() {
	// TODO Auto-generated destructor stub
}

vector<double> InputLayer::feedforward(vector<double> input) {
	vector<double> temp, output(neurons.size());

	double *deviceInput, *deviceOutput, deviceScalar;
	PassiveNeuron *deviceNeurons;

	// copy memory to device
	cudaMalloc((void **)&deviceInput, (input.size() * sizeof(double)));
	cudaMalloc((void **)&deviceOutput, (neurons.size() * sizeof(double)));
	cudaMalloc((void **)&deviceScalar, (sizeof(double)));
	cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(PassiveNeuron)));
	cudaMemcpy(&deviceInput[0], &input[0], (input.size() * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(&deviceScalar, &scalar, (sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(&deviceNeurons[0], &neurons[0], (neurons.size() * sizeof(PassiveNeuron)), cudaMemcpyHostToDevice);

	// start cuda kernel
	activateNeuron<<<dim3(1, 1), dim3(neurons.size(), 1)>>>(deviceInput, deviceNeurons, scalar, deviceOutput);

	// get the output from the device
	cudaMemcpy(&output[0], &deviceOutput[0], (neurons.size() * sizeof(double)), cudaMemcpyDeviceToHost);
	return output;
}

vector<double> InputLayer::backpropagate(vector<double> error, double learningRate) {
	return error;
}

