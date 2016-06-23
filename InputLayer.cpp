/*
 * InputLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "InputLayer.h"

InputLayer::InputLayer(int w, int d, double range) {
	// TODO Auto-generated constructor stub
	width = w;
	depth = d;
	scalar = 1 / range;
	for (int i = 0; i < width; i++) {
		neurons.push_back(new PassiveNeuron());
		cout << "Passive Neuron created " << i << endl;
	}  for (int i = 0; i < (w * d); i++) {
		synapses.push_back(new Synapse());
		cout << "Synapse created " << i << endl;
	}
}

InputLayer::~InputLayer() {
	// TODO Auto-generated destructor stub
}

vector<double> InputLayer::feedforward(vector<double> input) {
	vector<double> temp, output;

	// take the input and simply scale it within a range
	for (int i = 0; i < width; i++) {
		temp.push_back(((PassiveNeuron *)neurons[i])->get(input[i], scalar));
		cout << "Input Neuron " << i << " activating by " << temp[i] << endl;
	} for (int i = 0; i < depth; i++) {	// iterate through each synapse
		for (int j = 0; j < width; j++) {
			// each current neuron has as many synapses as there are neurons in next layer
			output.push_back(synapses[(j * depth) + i]->get(temp[j]));
			cout << "Synapse " << ((j * depth) + i) << " receiving " << temp[j] << " outputing " << output[output.size() - 1] << endl;
		}
	} return output;
}

vector<double> InputLayer::backpropagate(vector<double> error, double learningRate) {
	vector<double> sum;
		// iterate through each synapse connected to the next layer
		for (int i = 0; i < depth; i++) {
			double weightSum = 0;
			for (int j = 0; j < width; j++) weightSum += synapses[(j * depth) + i]->weight;
			for (int j = 0; j < width; j++) {
				if (i == 0) sum.push_back(0);
				// update the weight and bias variables
				synapses[(j * depth) + i]->weightedError = error[i] * synapses[(j * depth) + i]->weight / weightSum;
				synapses[(j * depth) + i]->weight -= learningRate * neurons[j]->activation * synapses[(j * depth) + i]->weightedError;
				synapses[(j * depth) + i]->bias -= learningRate * synapses[(j * depth) + i]->weightedError;

				// sum up the total weighted error for each neuron (potentially take the average)
				sum[j] += synapses[(j * depth) + i]->weightedError;
				if (i == (width - 1)) sum[j] /= depth;
			}
		} return sum;
}

