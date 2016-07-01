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

	double *deviceInput, *deviceOutput, *deviceScalar;
	PassiveNeuron *deviceNeurons;

	// copy memory to device
	if (cudaMalloc((void **)&deviceInput, (input.size() * sizeof(double))) != 0) cout << "error 1" << endl;
	if (cudaMalloc((void **)&deviceOutput, (neurons.size() * sizeof(double))) != 0) cout << "error 2" << endl;
	if (cudaMalloc((void **)&deviceScalar, sizeof(double)) != 0) cout << "error 3" << endl;
	if (cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(PassiveNeuron))) != 0) cout << "error 4" << endl;

	if (cudaMemcpy(&deviceInput[0], &input[0], (input.size() * sizeof(double)), cudaMemcpyHostToDevice) != 0) cout << "error 5" << endl;
	if (cudaMemcpy(&deviceScalar[0], &scalar, sizeof(double), cudaMemcpyHostToDevice) != 0) cout << "error 6" << endl;
	if (cudaMemcpy(&deviceNeurons[0], &neurons[0], (neurons.size() * sizeof(PassiveNeuron)), cudaMemcpyHostToDevice) != 0) cout << "error 7" << endl;

	// start cuda kernel
	cudaDeviceSynchronize();
	activateInputNeuron<<<dim3(1, 1), dim3(neurons.size(), 1)>>>(deviceInput, deviceNeurons, deviceScalar, deviceOutput);
	cudaDeviceSynchronize();

	// get the output from the device
	if (cudaMemcpy(&output[0], &deviceOutput[0], (neurons.size() * sizeof(double)), cudaMemcpyDeviceToHost) != 0) cout << "error__" << endl;
	cudaDeviceSynchronize();

	// release memory from GPU
	if (cudaFree(deviceInput) != 0) cout << "error 8" << endl;
	if (cudaFree(deviceOutput) != 0) cout << "error 9" << endl;
	if (cudaFree(deviceScalar) != 0) cout << "error 10" << endl;
	if (cudaFree(deviceNeurons) != 0) cout << "error** 11" << endl;

	return output;
}

vector<double> InputLayer::backpropagate(vector<double> error, double learningRate) {
	return error;
}

