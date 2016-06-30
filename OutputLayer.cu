/*
 * OutputLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "OutputLayer.cuh"

__global__ void activateNeuron(double *input, Neuron nodes[], double *output) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;	// the current layer neurons
	output[idx] = (nodes[idx]).get(input[idx]);
}

__global__ void activateSynapse(double *input, Synapse connections[], double *output) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;	// the current layer
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;	// the previous layer
	output[(idx * (gridDim.y * blockDim.y)) + idy] = (connections[(idx * (gridDim.y * blockDim.y)) + idy]).get(input[idy]);
}

__global__ void sumInputFromSynapse(double *input, double *output) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;	// the current layer
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;	// the previous layer
	output[idx] += input[(idx * (gridDim.y * blockDim.y)) + idy];
}

__global__ void gradientDescent(double *error, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[]) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;	// the current layer
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;	// the previous layer
	connections[(idx * (gridDim.y * blockDim.y)) + idy].weight -= learningRate * error[idx] * nodes[idx].derivative * previous[idy].activation;
	//connections[(idx * (gridDim.y * blockDim.y)) + idy].bias -= learningRate * error[idx] * nodes[idx].derivative;
}

__global__ void sumWeightedError(double *error, Neuron nodes[], Synapse connections[], double *output) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;	// the current layer
	int idy = (blockIdx.y * blockDim.y) + threadIdx.y;	// the previous layer
	output[idy] += ((error[idx] * nodes[idx].derivative) * connections[(idx * (gridDim.y * blockDim.y)) + idy].weight) + connections[(idx * (gridDim.y * blockDim.y)) + idy].bias;
}


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
	vector<double> sum(currentLayerNeurons), output(synapses.size());	// variables to store data for math operations

	double *deviceInput, *deviceOutput, *deviceSum, *deviceActivation;
	Synapse *deviceSynapses;
	Neuron *deviceNeurons;

	// copy memory to device
	cudaMalloc((void **)&deviceInput, (input.size() * sizeof(double)));
	cudaMalloc((void **)&deviceOutput, (synapses.size() * sizeof(double)));
	cudaMalloc((void **)&deviceSum, (neurons.size() * sizeof(double)));
	cudaMalloc((void **)&deviceActivation, (neurons.size() * sizeof(double)));
	cudaMalloc((void **)&deviceSynapses, (synapses.size() * sizeof(Synapse)));
	cudaMalloc((void **)&deviceNeurons, (neurons.size() * sizeof(Neuron)));
	cudaMemcpy(deviceInput, &input[0], (input.size() * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSynapses, &synapses[0], (synapses.size() * sizeof(Synapse)), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceNeurons, &neurons[0], (neurons.size() * sizeof(Neuron)), cudaMemcpyHostToDevice);
	cudaMemset(deviceSum, 0, (neurons.size() * sizeof(double)));

	activateSynapse<<<dim3(1, 1), dim3(currentLayerNeurons, previousLayerNeurons)>>>(deviceInput, deviceSynapses, deviceOutput);
	sumInputFromSynapse<<<dim3(1, 1), dim3(currentLayerNeurons, previousLayerNeurons)>>>(deviceOutput, deviceSum);
	activateNeuron<<<dim3(1, 1), dim3(currentLayerNeurons, 1)>>>(deviceSum, deviceNeurons, deviceActivation);

	// get the output from the device
	cudaMemcpy(&output[0], &deviceActivation[0],(neurons.size() * sizeof(double)), cudaMemcpyDeviceToHost);
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
	cudaMalloc((void **)&(&deviceLearningRate), sizeof(double));	// this may be a problem since is only one variable not array
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
