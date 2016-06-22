/*
 * OutputLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "OutputLayer.h"

OutputLayer::OutputLayer(int w) {
	// TODO Auto-generated constructor stub
	width = w;

	// add neurons to this layer
	for (int i = 0; i < w; i++) {
		neurons.push_back(new Neuron());
		cout << "Neuron created " << i << endl;
	}
}

OutputLayer::~OutputLayer() {
	// TODO Auto-generated destructor stub
}

vector<double> OutputLayer::feedforward(vector<double> input) {
	vector<double> sum, output;	// variables to store data for math operations
	for (int i = 0; i < width; i++) {	// iterate through each synapse for input
		sum.push_back(0);
		for (unsigned int j = 0; j < (input.size() / width); j++) {
			sum[i] += input[(i * (input.size() / width)) + j];
		}
		output.push_back(neurons[i]->get(sum[i]));
		cout << "Neuron " << i << " activating by " << output[i] << endl;
	} return output;
}

vector<double> OutputLayer::backpropagate(vector<double> error, double learningRate) {
	// output layer has no synapses to update
	return error;
}
