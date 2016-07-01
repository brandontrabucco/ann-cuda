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
	vector<double> sum(currentLayerNeurons), output(currentLayerNeurons);	// variables to store data for math operations

	double *deviceInput, *deviceOutput, *deviceSum, *deviceActivation;
	Synapse *deviceSynapses;
	Neuron *deviceNeurons;

	// copy memory to device
	if (cudaMalloc((void **)&deviceInput, (input.size() * sizeof(double))) != 0) cout << "error 1" << endl;
	if (cudaMalloc((void **)&deviceOutput, (synapses.size() * sizeof(double))) != 0) cout << "error 2" << endl;
	if (cudaMalloc((void **)&deviceSum, (neurons.size() * sizeof(double))) != 0) cout << "error 3" << endl;
	if (cudaMalloc((void **)&deviceActivation, (neurons.size() * sizeof(double))) != 0) cout << "error 4" << endl;
	if (cudaMalloc((void **)&deviceSynapses, (synapses.size() * sizeof(Synapse))) != 0) cout << "error 5" << endl;
	if (cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(Neuron))) != 0) cout << "error 6" << endl;

	if (cudaMemcpy(deviceInput, &input[0], (input.size() * sizeof(double)), cudaMemcpyHostToDevice) != 0) cout << "error 7" << endl;
	if (cudaMemcpy(deviceSynapses, &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice) != 0) cout << "error 8" << endl;
	if (cudaMemcpy(deviceNeurons, &neurons[0], (neurons.size() * sizeof(Neuron)), cudaMemcpyHostToDevice) != 0) cout << "error 9" << endl;
	if (cudaMemset(deviceSum, 0, (neurons.size() * sizeof(double))) != 0) cout << "error 10" << endl;

	activateSynapse<<<dim3(1, 1), dim3(currentLayerNeurons, previousLayerNeurons)>>>(deviceInput, deviceSynapses, deviceOutput);
	sumInputFromSynapse<<<dim3(1, 1), dim3(currentLayerNeurons, previousLayerNeurons)>>>(deviceOutput, deviceSum);
	activateNeuron<<<dim3(1, 1), dim3(currentLayerNeurons, 1)>>>(deviceSum, deviceNeurons, deviceActivation);

	// get the output from the device
	if (cudaMemcpy(&output[0], &deviceActivation[0],(neurons.size() * sizeof(double)), cudaMemcpyDeviceToHost) != 0) cout << "error 11" << endl;
	return output;
}

vector<double> OutputLayer::backpropagate(vector<double> error, double learningRate, vector<Neuron> previousLayer) {
	// iterate through each synapse connected to the previous layer
	vector<double> sum(previousLayerNeurons);

	double *deviceError, *deviceSum, deviceLearningRate;
	Synapse *deviceSynapses;
	Neuron *deviceNeurons, *devicePreviousLayer;

	cudaMalloc((void **)&deviceError, (error.size() * sizeof(double)));
	cudaMalloc((void **)&deviceSum, (previousLayerNeurons * sizeof(double)));
	cudaMalloc((void **)&deviceLearningRate, sizeof(double));	// this may be a problem since is only one variable not array
	cudaMalloc((void **)&deviceSynapses, (synapses.size() * sizeof(Synapse)));
	cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(Neuron)));
	cudaMalloc((void **)&devicePreviousLayer, (previousLayer.size() * sizeof(Neuron)));

	cudaMemcpy(deviceError, &error[0], (error.size() * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(&deviceLearningRate, &learningRate, (error.size() * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSynapses, &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceNeurons, &neurons[0], (neurons.size() * sizeof(Neuron)), cudaMemcpyHostToDevice);
	cudaMemcpy(devicePreviousLayer, &previousLayer[0], (previousLayer.size() * sizeof(Neuron)), cudaMemcpyHostToDevice);
	cudaMemset(deviceSum, 0, (previousLayerNeurons * sizeof(double)));

	gradientDescent<<<dim3(1, 1), dim3(currentLayerNeurons, previousLayerNeurons)>>>(deviceError, deviceLearningRate, deviceNeurons, devicePreviousLayer, deviceSynapses);
	sumWeightedError<<<dim3(1, 1), dim3(currentLayerNeurons, previousLayerNeurons)>>>(deviceError, deviceNeurons, deviceSynapses, deviceSum);

	cudaMemcpy(&synapses[0], &deviceSynapses[0],(synapses.size() * sizeof(Synapse)), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sum[0], &deviceSum[0],(previousLayerNeurons * sizeof(double)), cudaMemcpyDeviceToHost);
	return sum;
}
