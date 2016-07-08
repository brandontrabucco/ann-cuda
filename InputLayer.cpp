/*
 * InputLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "InputLayer.h"

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
	vector<double> temp, output;

	// take the input and simply scale it within a range
	for (int i = 0; i < currentLayerNeurons; i++) {
		temp.push_back(((PassiveNeuron&)(neurons[i])).get(input[i], scalar));
		if (debug) cout << "Neuron " << neurons[i].index << " activating by " << temp[i] << " from " << input[i] << endl;
	} return temp;
}

vector<double> InputLayer::backpropagate(vector<double> error, double learningRate) {
	return error;
}

