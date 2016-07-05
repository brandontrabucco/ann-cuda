/*
 * HiddenLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "HiddenLayer.cuh"

HiddenLayer::HiddenLayer(int w, int d, bool db) {
	// TODO Auto-generated constructor stub
	debug = db;
	currentLayerNeurons = w;
	previousLayerNeurons = d;

	vector<int> temp = factor(currentLayerNeurons);
	kernelGridHeight = temp[0];
	kernelGridWidth = temp[1];

	temp = factor(previousLayerNeurons);
	kernelBlockHeight = temp[0];
	kernelBlockWidth = temp[1];

	// add neurons and synapses to this layer
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

HiddenLayer::~HiddenLayer() {
	// TODO Auto-generated destructor stub
}

// parallelize each synapse and neuron

vector<double> HiddenLayer::feedforward(vector<double> input) {
	vector<double> output(currentLayerNeurons);	// variables to store data for math operations

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

	if (cudaMemcpy(&deviceInput[0], &input[0], (input.size() * sizeof(double)), cudaMemcpyHostToDevice) != 0) cout << "error 7" << endl;
	if (cudaMemcpy(&deviceSynapses[0], &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice) != 0) cout << "error 8" << endl;
	if (cudaMemcpy(&deviceNeurons[0], &neurons[0], (neurons.size() * sizeof(Neuron)), cudaMemcpyHostToDevice) != 0) cout << "error 9" << endl;
	if (cudaMemset(&deviceSum[0], 0, (neurons.size() * sizeof(double))) != 0) cout << "error 10" << endl;

	cudaDeviceSynchronize();
	activateSynapse<<<dim3(kernelGridWidth, kernelGridHeight), dim3(kernelBlockWidth, kernelBlockHeight)>>>(deviceInput, deviceSynapses, deviceOutput);	// a block represents current layer, thread is previous layer
	cudaDeviceSynchronize();
	sumInputFromSynapse<<<dim3(1, 1), dim3(kernelGridWidth, kernelGridHeight)>>>(deviceOutput, deviceSum, previousLayerNeurons);
	cudaDeviceSynchronize();
	activateNeuron<<<dim3(1, 1), dim3(kernelGridWidth, kernelGridHeight)>>>(deviceSum, deviceNeurons, deviceActivation);
	cudaDeviceSynchronize();


	// get the output from the device
	if (cudaMemcpy(&output[0], &deviceActivation[0],(neurons.size() * sizeof(double)), cudaMemcpyDeviceToHost) != 0) cout << "error __ 11" << endl;
	cudaDeviceSynchronize();

	// release memory from GPU
	if (cudaFree(deviceInput) != 0) cout << "error 12" << endl;
	if (cudaFree(deviceOutput) != 0) cout << "error 13" << endl;
	if (cudaFree(deviceSum) != 0) cout << "error 14" << endl;
	if (cudaFree(deviceActivation) != 0) cout << "error 15" << endl;
	if (cudaFree(deviceSynapses) != 0) cout << "error 16" << endl;
	if (cudaFree(deviceNeurons) != 0) cout << "error 17" << endl;
	cudaDeviceSynchronize();

	return output;
}

// parallelize each synapse update

vector<double> HiddenLayer::backpropagate(vector<double> error, double learningRate, vector<Neuron> previousLayer) {
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
	cudaDeviceSynchronize();

	cudaMemcpy(deviceError, &error[0], (error.size() * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(&deviceLearningRate, &learningRate, (error.size() * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSynapses, &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceNeurons, &neurons[0], (neurons.size() * sizeof(Neuron)), cudaMemcpyHostToDevice);
	cudaMemcpy(devicePreviousLayer, &previousLayer[0], (previousLayer.size() * sizeof(Neuron)), cudaMemcpyHostToDevice);
	cudaMemset(deviceSum, 0, (previousLayerNeurons * sizeof(double)));
	cudaDeviceSynchronize();

	gradientDescent<<<dim3(kernelGridWidth, kernelGridHeight), dim3(kernelBlockWidth, kernelBlockHeight)>>>(deviceError, deviceLearningRate, deviceNeurons, devicePreviousLayer, deviceSynapses);
	cudaDeviceSynchronize();
	sumWeightedError<<<dim3(1, 1), dim3(kernelGridWidth, kernelGridHeight)>>>(deviceError, deviceNeurons, deviceSynapses, deviceSum, previousLayerNeurons);

	cudaDeviceSynchronize();
	cudaMemcpy(&synapses[0], &deviceSynapses[0],(synapses.size() * sizeof(Synapse)), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sum[0], &deviceSum[0],(previousLayerNeurons * sizeof(double)), cudaMemcpyDeviceToHost);
	return sum;
}

