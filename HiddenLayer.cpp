/*
 * HiddenLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(int w, int d) {
	// TODO Auto-generated constructor stub
	width = w;
	depth = d;

	// add neurons and synapses to this layer
	for (int i = 0; i < w; i++) {
		neurons.push_back(new Neuron());
		cout << "Neuron created " << i << endl;
	} for (int i = 0; i < (w * d); i++) {
		synapses.push_back(new Synapse());
		cout << "Synapse created " << i << endl;
	}
}

HiddenLayer::~HiddenLayer() {
	// TODO Auto-generated destructor stub
}

vector<double> HiddenLayer::feedforward(vector<double> input) {
	vector<double> temp, sum, output;	// variables to store data for math operations
	for (int i = 0; i < width; i++) {	// iterate through each synapse for input
		sum.push_back(0);
		for (unsigned int j = 0; j < (input.size() / width); j++) {
			sum[i] += input[(i * (input.size() / width)) + j];
		}
	} for (int i = 0; i < width; i++) {
		// data is scaled, aligned, and summed
		// compute current layer neural activation
		temp.push_back(neurons[i]->get(sum[i]));
		cout << "Neuron " << i << " activating by " << temp[i] << endl;
	} for (int i = 0; i < depth; i++) {	// iterate through each synapse
		for (int j = 0; j < width; j++) {
			// each current neuron has as many synapses as there are neurons in next layer
			output.push_back(synapses[(j * depth) + i]->get(temp[j]));
			cout << "Synapse " << ((j * depth) + i) << " receiving " << temp[j] << " outputing " << output[output.size() - 1] << endl;
		}
	} return output;
}

vector<double> HiddenLayer::backpropagate(vector<double> error, double learningRate) {
	vector<double> sum;
	// iterate through each synapse connected to the next layer
	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < width; j++) {
			if (i == 0) sum.push_back(0);
			// update the weight and bias variables (need to take the weighted error in proportion to the sum of weights to a neuron)
			synapses[(j * depth) + i]->weightedError = error[i] * synapses[(j * depth) + i]->weight;
			synapses[(j * depth) + i]->weight -= learningRate * neurons[j]->activation * synapses[(j * depth) + i]->weightedError;
			synapses[(j * depth) + i]->bias -= learningRate * synapses[(j * depth) + i]->weightedError;

			// sum up the total weighted error for each neuron (potentially take the average)
			sum[j] += synapses[(j * depth) + i]->weightedError;
			if (i == (width - 1)) sum[j] /= depth;
		}
	} return sum;
}

