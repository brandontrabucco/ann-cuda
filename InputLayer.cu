/*
 * InputLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "InputLayer.cuh"

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
	vector<double> output(neurons.size());

	double *deviceInput, *deviceOutput;
	PassiveNeuron *deviceNeurons;

	// copy memory to device
	int status;
	if ((status = cudaMalloc((void **)&deviceInput, (input.size() * sizeof(double)))) != 0) cout << "error h-1 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceOutput, (neurons.size() * sizeof(double)))) != 0) cout << "error h-2 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(PassiveNeuron)))) != 0) cout << "error h-4 " << status << endl;

	if ((status = cudaMemcpy(&deviceInput[0], &input[0], (input.size() * sizeof(double)), cudaMemcpyHostToDevice)) != 0) cout << "error h-5 " << status << endl;
	if ((status = cudaMemcpy(&deviceNeurons[0], &neurons[0], (neurons.size() * sizeof(PassiveNeuron)), cudaMemcpyHostToDevice)) != 0) cout << "error h-7 " << status << endl;

	// start cuda kernel
	KernelAdapter::startInputNeuronKernel(deviceInput, deviceNeurons, scalar, deviceOutput, currentLayerNeurons);

	// get the output from the device
	if ((status = cudaMemcpy(&output[0], &deviceOutput[0], (neurons.size() * sizeof(double)), cudaMemcpyDeviceToHost)) != 0) cout << "error__ " << status << endl;
	if ((status = cudaMemcpy(&neurons[0], &deviceNeurons[0],(neurons.size() * sizeof(PassiveNeuron)), cudaMemcpyDeviceToHost)) != 0) cout << "error _*_ h-" << status << endl;
	cudaDeviceSynchronize();

	// release memory from GPU
	if ((status = cudaFree(deviceInput)) != 0) cout << "error h-8 " << status << endl;
	if ((status = cudaFree(deviceOutput)) != 0) cout << "error h-9 " << status << endl;
	if ((status = cudaFree(deviceNeurons)) != 0) cout << "error** h-10 " << status << endl;
	cudaDeviceSynchronize();

	return output;
}

vector<double> InputLayer::backpropagate(vector<double> error, double learningRate) {
	return error;
}

