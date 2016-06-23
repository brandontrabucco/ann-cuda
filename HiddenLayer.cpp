/*
 * HiddenLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(int w, int d, bool db) {
	// TODO Auto-generated constructor stub
	debug = db;
	width = w;
	depth = d;

	// add neurons and synapses to this layer
	for (int i = 0; i < w; i++) {
		Neuron *n = new Neuron();
		n->index = i;
		neurons.push_back(n);
		if (debug) cout << "Neuron created " << i << endl;
	} for (int i = 0; i < (w * d); i++) {
		Synapse *s = new Synapse();
		s->index = i;
		synapses.push_back(s);
		if (debug) cout << "Synapse created " << i << endl;
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
			if (debug) cout << "Neuron " << i << " summing index " << (i * (input.size() / width)) + j << endl;
		}
	} for (int i = 0; i < width; i++) {
		// data is scaled, aligned, and summed
		// compute current layer neural activation
		temp.push_back(neurons[i]->get(sum[i]));
		if (debug) cout << "Neuron " << neurons[i]->index << " activating at " << i << endl;
	} for (int i = 0; i < depth; i++) {	// iterate through each synapse
		for (int j = 0; j < width; j++) {
			// each current neuron has as many synapses as there are neurons in next layer
			output.push_back(synapses[(j * depth) + i]->get(temp[j]));
			if (debug) cout << "Synapse " << synapses[(j * depth) + i]->index << " activating at " << ((j * depth) + i) << " index " << ((i * width) + j) << endl;
		}
	} return output;
}

vector<double> HiddenLayer::backpropagate(vector<double> errorPrime, double learningRate) {
	vector<double> sum;
	// iterate through each synapse connected to the next layer
	for (int i = 0; i < depth; i++) {
		double weightSum = 0;
		for (int j = 0; j < width; j++) weightSum += synapses[(j * depth) + i]->weight;
		for (int j = 0; j < width; j++) {
			if (i == 0) sum.push_back(0);
			// update the weight and bias variables (need to take the weighted error in proportion to the sum of weights to a neuron)
			synapses[(j * depth) + i]->weightedErrorPrime = errorPrime[i] * synapses[(j * depth) + i]->weight / weightSum;
			synapses[(j * depth) + i]->weight -= learningRate * neurons[j]->activation * synapses[(j * depth) + i]->weightedErrorPrime;
			synapses[(j * depth) + i]->bias -= learningRate * synapses[(j * depth) + i]->weightedErrorPrime;

			if (debug) cout << "Synapse " << synapses[(j * depth) + i]->index << " updating at " << ((j * depth) + i) << endl;

			// sum up the total weighted error for each neuron (potentially take the average)
			sum[j] += synapses[(j * depth) + i]->weightedErrorPrime;
			//if (i == (width - 1)) sum[j] /= depth;
		}
	} return sum;
}

