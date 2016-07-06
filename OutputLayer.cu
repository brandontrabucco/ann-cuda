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

	vector<int> temp = factor(currentLayerNeurons);
	kernelGridHeight = temp[0];
	kernelGridWidth = temp[1];

	temp = factor(previousLayerNeurons);
	kernelBlockHeight = temp[0];
	kernelBlockWidth = temp[1];

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
	if ((status = cudaMalloc((void **)&deviceInput, (input.size() * sizeof(double)))) != 0) cout << "error 1 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceOutput, (synapses.size() * sizeof(double)))) != 0) cout << "error 2 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceSum, (neurons.size() * sizeof(double)))) != 0) cout << "error 3 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceActivation, (neurons.size() * sizeof(double)))) != 0) cout << "error 4 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceSynapses, (synapses.size() * sizeof(Synapse)))) != 0) cout << "error 5 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(Neuron)))) != 0) cout << "error 6 " << status << endl;

	if ((status = cudaMemcpy(&deviceInput[0], &input[0], (input.size() * sizeof(double)), cudaMemcpyHostToDevice)) != 0) cout << "error 7 " << status << endl;
	if ((status = cudaMemcpy(&deviceSynapses[0], &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice)) != 0) cout << "error 8 " << status << endl;
	if ((status = cudaMemcpy(&deviceNeurons[0], &neurons[0], (neurons.size() * sizeof(Neuron)), cudaMemcpyHostToDevice)) != 0) cout << "error 9 " << status << endl;
	if ((status = cudaMemset(&deviceSum[0], 0, (neurons.size() * sizeof(double)))) != 0) cout << "error 10 " << status << endl;

	cudaDeviceSynchronize();
	activateSynapse<<<dim3(kernelGridWidth, kernelGridHeight), dim3(kernelBlockWidth, kernelBlockHeight)>>>(deviceInput, deviceSynapses, deviceOutput);	// a block represents current layer, thread is previous layer
	cudaDeviceSynchronize();
	sumInputFromSynapse<<<dim3(1, 1), dim3(kernelGridWidth, kernelGridHeight)>>>(deviceOutput, deviceSum, previousLayerNeurons);
	cudaDeviceSynchronize();
	activateNeuron<<<dim3(1, 1), dim3(kernelGridWidth, kernelGridHeight)>>>(deviceSum, deviceNeurons, deviceActivation);
	cudaDeviceSynchronize();

	// get the output from the device
	if ((status = cudaMemcpy(&output[0], &deviceActivation[0],(neurons.size() * sizeof(double)), cudaMemcpyDeviceToHost)) != 0) cout << "error 11 " << status << endl;
	if ((status = cudaMemcpy(&neurons[0], &deviceNeurons[0],(neurons.size() * sizeof(Neuron)), cudaMemcpyDeviceToHost)) != 0) cout << "error 12 " << status << endl;
	cudaDeviceSynchronize();

	// release memory from GPU
	if ((status = cudaFree(deviceInput)) != 0) cout << "error 13 " << status << endl;
	if ((status = cudaFree(deviceOutput)) != 0) cout << "error 14 " << status << endl;
	if ((status = cudaFree(deviceSum)) != 0) cout << "error 15 " << status << endl;
	if ((status = cudaFree(deviceActivation)) != 0) cout << "error 16 " << status << endl;
	if ((status = cudaFree(deviceSynapses)) != 0) cout << "error 17 " << status << endl;
	if ((status = cudaFree(deviceNeurons)) != 0) cout << "error 18 " << status << endl;
	cudaDeviceSynchronize();

	return output;
}

vector<double> OutputLayer::backpropagate(vector<double> error, double learningRate, vector<Neuron> previousLayer) {
	// iterate through each synapse connected to the previous layer
	vector<double> sum(previousLayerNeurons);	// must be initialized to be read and write

	double *deviceError, *deviceSum;
	Synapse *deviceSynapses;
	Neuron *deviceNeurons, *devicePreviousLayer;

	int status;
	if ((status = cudaMalloc((void **)&deviceError, (error.size() * sizeof(double)))) != 0) cout << "error 1 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceSum, (previousLayerNeurons * sizeof(double)))) != 0) cout << "error 2 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceSynapses, (synapses.size() * sizeof(Synapse)))) != 0) cout << "error 4 " << status << endl;
	if ((status = cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(Neuron)))) != 0) cout << "error 5 " << status << endl;
	if ((status = cudaMalloc((void **)&devicePreviousLayer, (previousLayer.size() * sizeof(Neuron)))) != 0) cout << "error 6 " << status << endl;

	if ((status = cudaMemcpy(&deviceError[0], &error[0], (error.size() * sizeof(double)), cudaMemcpyHostToDevice)) != 0) cout << "error 7 " << status << endl;
	if ((status = cudaMemcpy(&deviceSynapses[0], &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice)) != 0) cout << "error 9 " << status << endl;
	if ((status = cudaMemcpy(&deviceNeurons[0], &neurons[0], (neurons.size() * sizeof(Neuron)), cudaMemcpyHostToDevice)) != 0) cout << "error 10 " << status << endl;
	if ((status = cudaMemcpy(&devicePreviousLayer[0], &previousLayer[0], (previousLayer.size() * sizeof(Neuron)), cudaMemcpyHostToDevice)) != 0) cout << "error 11 " << status << endl;
	if ((status = cudaMemset(&deviceSum[0], 0, (previousLayerNeurons * sizeof(double))) != 0)) cout << "error 12 " << status << endl;
	cudaDeviceSynchronize();

	gradientDescent<<<dim3(kernelGridWidth, kernelGridHeight), dim3(kernelBlockWidth, kernelBlockHeight)>>>(deviceError, learningRate, deviceNeurons, devicePreviousLayer, deviceSynapses);
	cudaDeviceSynchronize();
	// iterate for each neuron sum in previous layer;
	sumWeightedError<<<dim3(1, 1), dim3(kernelBlockWidth, kernelBlockHeight)>>>(deviceError, deviceNeurons, deviceSynapses, deviceSum, (currentLayerNeurons));

	cudaDeviceSynchronize();
	if ((status = cudaMemcpy(&synapses[0], &deviceSynapses[0],(synapses.size() * sizeof(Synapse)), cudaMemcpyDeviceToHost)) != 0) cout << "error 13 " << status << endl;
	if ((status = cudaMemcpy(&sum[0], &deviceSum[0],(previousLayerNeurons * sizeof(double)), cudaMemcpyDeviceToHost)) != 0) cout << "error 14 " << status << endl;
	cudaDeviceSynchronize();

	// release memory from GPU
	if ((status = cudaFree(deviceError)) != 0) cout << "error 15 " << status << endl;
	if ((status = cudaFree(deviceSum)) != 0) cout << "error 16 " << status << endl;
	if ((status = cudaFree(deviceSynapses)) != 0) cout << "error 17 " << status << endl;
	if ((status = cudaFree(deviceNeurons)) != 0) cout << "error 18 " << status << endl;
	if ((status = cudaFree(devicePreviousLayer)) != 0) cout << "error 19 " << status << endl;
	cudaDeviceSynchronize();

	return sum;
}
