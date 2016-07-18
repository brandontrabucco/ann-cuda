/*
 * OutputLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "OutputLayer.cuh"

OutputLayer::OutputLayer(int w, int d, bool db) {
	// TODO Auto-generated constructor stub
	debug = db;
	currentLayerNeurons = w;
	previousLayerNeurons = d;

	// add neurons to this layer
	for (int i = 0; i < w; i++) {
		Neuron n = Neuron();
		n.index = i;
		neurons.push_back(n);
		if (debug) cout << "Neuron created " << i << endl;
	} for (int i = 0; i < (w * d); i++) {
		Synapse s = Synapse();
		s.index = i;
		synapses.push_back(s);
		if (debug) cout << "Synapse created " << i << endl;
	}
}

OutputLayer::~OutputLayer() {
	// TODO Auto-generated destructor stub
}

vector<double> OutputLayer::feedforward(vector<double> input) {
	vector<double> output(currentLayerNeurons);	// variables to store data for math operations

	double *deviceInput, *deviceOutput, *deviceSum, *deviceActivation;
	Synapse *deviceSynapses;
	Neuron *deviceNeurons;

	// copy memory to device
	int status;
	long long startTime = getNanoSec();
	if ((status = cudaMalloc((void **)&deviceInput, (input.size() * sizeof(double)))) != 0) cout << "error o-1 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceOutput, (synapses.size() * sizeof(double)))) != 0) cout << "error o-2 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceSum, (neurons.size() * sizeof(double)))) != 0) cout << "error o-3 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceActivation, (neurons.size() * sizeof(double)))) != 0) cout << "error o-4 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceSynapses, (synapses.size() * sizeof(Synapse)))) != 0) cout << "error o-5 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(Neuron)))) != 0) cout << "error o-6 " << status << endl;

	if ((status = cudaMemcpy(&deviceInput[0], &input[0], (input.size() * sizeof(double)), cudaMemcpyHostToDevice)) != 0) cout << "error o-7 " << status << endl;
	if ((status = cudaMemcpy(&deviceSynapses[0], &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice)) != 0) cout << "error o-8 " << status << endl;
	if ((status = cudaMemcpy(&deviceNeurons[0], &neurons[0], (neurons.size() * sizeof(Neuron)), cudaMemcpyHostToDevice)) != 0) cout << "error o-9 " << status << endl;
	if ((status = cudaMemset(&deviceSum[0], 0, (neurons.size() * sizeof(double)))) != 0) cout << "error o-10 " << status << endl;
	NeuralNetwork::overhead += (getNanoSec() - startTime);

	startTime = getNanoSec();
	KernelAdapter::startSynapseKernel(deviceInput, deviceSynapses, deviceOutput, currentLayerNeurons, previousLayerNeurons);
	KernelAdapter::startSumInputKernel(deviceOutput, deviceSum, currentLayerNeurons, previousLayerNeurons);
	KernelAdapter::startNeuronKernel(deviceSum, deviceNeurons, deviceActivation, currentLayerNeurons);
	NeuralNetwork::computation += (getNanoSec() - startTime);

	// get the output from the device
	startTime = getNanoSec();
	if ((status = cudaMemcpy(&output[0], &deviceActivation[0],(neurons.size() * sizeof(double)), cudaMemcpyDeviceToHost)) != 0) cout << "error o-11 " << status << endl;
	if ((status = cudaMemcpy(&neurons[0], &deviceNeurons[0],(neurons.size() * sizeof(Neuron)), cudaMemcpyDeviceToHost)) != 0) cout << "error o-12 " << status << endl;
	cudaDeviceSynchronize();

	// release memory from GPU
	if ((status = cudaFree(deviceInput)) != 0) cout << "error o-13 " << status << endl;
	if ((status = cudaFree(deviceOutput)) != 0) cout << "error o-14 " << status << endl;
	if ((status = cudaFree(deviceSum)) != 0) cout << "error o-15 " << status << endl;
	if ((status = cudaFree(deviceActivation)) != 0) cout << "error o-16 " << status << endl;
	if ((status = cudaFree(deviceSynapses)) != 0) cout << "error o-17 " << status << endl;
	if ((status = cudaFree(deviceNeurons)) != 0) cout << "error o-18 " << status << endl;
	cudaDeviceSynchronize();
	NeuralNetwork::overhead += (getNanoSec() - startTime);

	return output;
}

vector<double> OutputLayer::backpropagate(vector<double> error, double learningRate, vector<Neuron> previousLayer) {
	// iterate through each synapse connected to the previous layer
	vector<double> sum(previousLayerNeurons);	// must be initialized to be read and write

	double *deviceError, *deviceSum;
	Synapse *deviceSynapses;
	Neuron *devicePreviousLayer;

	int status;
	long long startTime = getNanoSec();
	if ((status = cudaMalloc((void **)&deviceError, (error.size() * sizeof(double)))) != 0) cout << "error o-1 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceSum, (previousLayerNeurons * sizeof(double)))) != 0) cout << "error o-2 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceSynapses, (synapses.size() * sizeof(Synapse)))) != 0) cout << "error o-4 " << status << endl;
	if ((status = cudaMalloc((void **)&devicePreviousLayer, (previousLayer.size() * sizeof(Neuron)))) != 0) cout << "error o-6 " << status << endl;

	if ((status = cudaMemcpy(&deviceError[0], &error[0], (error.size() * sizeof(double)), cudaMemcpyHostToDevice)) != 0) cout << "error o-7 " << status << endl;
	if ((status = cudaMemcpy(&deviceSynapses[0], &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice)) != 0) cout << "error o-9 " << status << endl;
	if ((status = cudaMemcpy(&devicePreviousLayer[0], &previousLayer[0], (previousLayer.size() * sizeof(Neuron)), cudaMemcpyHostToDevice)) != 0) cout << "error o-11 " << status << endl;
	if ((status = cudaMemset(&deviceSum[0], 0, (previousLayerNeurons * sizeof(double))) != 0)) cout << "error o-12 " << status << endl;
	NeuralNetwork::overhead += (getNanoSec() - startTime);

	startTime = getNanoSec();
	KernelAdapter::startOutputLayerGradientDescentKernel(deviceError, learningRate, devicePreviousLayer, deviceSynapses, currentLayerNeurons, previousLayerNeurons);
	KernelAdapter::startOutputLayerSumErrorKernel(deviceError, deviceSynapses, deviceSum, currentLayerNeurons, previousLayerNeurons);
	NeuralNetwork::computation += (getNanoSec() - startTime);

	// get output from device
	startTime = getNanoSec();
	if ((status = cudaMemcpy(&synapses[0], &deviceSynapses[0],(synapses.size() * sizeof(Synapse)), cudaMemcpyDeviceToHost)) != 0) cout << "error o-13 " << status << endl;
	if ((status = cudaMemcpy(&sum[0], &deviceSum[0],(previousLayerNeurons * sizeof(double)), cudaMemcpyDeviceToHost)) != 0)cout << "error o-14 " << status << endl;
	cudaDeviceSynchronize();

	// release memory from GPU
	if ((status = cudaFree(deviceError)) != 0) cout << "error test o-15 " << status << endl;
	if ((status = cudaFree(deviceSum)) != 0)cout << "error o-16 " << status << endl;
	if ((status = cudaFree(deviceSynapses)) != 0) cout << "error o-17 " << status << endl;
	if ((status = cudaFree(devicePreviousLayer)) != 0) cout << "error o-19 " << status << endl;
	cudaDeviceSynchronize();
	NeuralNetwork::overhead += (getNanoSec() - startTime);

	return sum;
}
