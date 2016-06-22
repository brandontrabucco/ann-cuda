/*
 * Network.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "Network.h"

Network::Network(vector<int> size, double range, double rate) {
	// TODO Auto-generated constructor stub
	learningRate = rate;
	for (unsigned int i = 0; i < size.size(); i++) {
		if (i == 0) layers.push_back(new InputLayer(size[i], size[i + 1], range));
		else if (i == (size.size() - 1)) layers.push_back(new OutputLayer(size[i]));
		else if (i > 0 && (i < (size.size() - 1))) layers.push_back(new HiddenLayer(size[i], size[i + 1]));
	}
}

Network::~Network() {
	// TODO Auto-generated destructor stub
}

void Network::classify(double input) {
	vector<double> temp;
	for (unsigned int i = 0; i < layers[0]->neurons.size(); i++) temp.push_back(input);
	feedforward(temp);
}

vector<double> Network::feedforward(vector<double> input) {
	vector<double> temp;
	temp = input;
	for (unsigned int i = 0; i < layers.size(); i++) {
		// feed the input through each layer of network
		cout << endl << "Layer " << i << " propagating" << endl << endl;
		if (i == 0) temp = ((InputLayer *)layers[i])->feedforward(temp);
		else if (i == (layers.size() - 1)) temp = ((OutputLayer *)layers[i])->feedforward(temp);
		else if (i > 0 && (i < (layers.size() - 1))) temp = ((HiddenLayer *)layers[i])->feedforward(temp);
		cout << endl << "Layer " << i << " finished" << endl << endl;
	} for (unsigned int i = 0; i < layers[layers.size() - 1]->neurons.size(); i++) {
		// show the output of the last layer
		cout << "Output of Neuron " << i << " : " << temp[i] << endl;
	}
	return temp;
}

void Network::train(double input, double actual) {
	vector<double> temp, output, error;
	for (unsigned int i = 0; i < layers[0]->neurons.size(); i++) temp.push_back(input);
	output = feedforward(temp);

	// get error with respect to each of the output nodes
	for (unsigned int i = 0; i < output.size(); i++) {
		error.push_back(output[i] - actual);
		cout << "Output Neuron " << i << " error " << error[i] << endl;
	}
	backpropagate(error);
}

void Network::backpropagate(vector<double> error) {
	vector<double>  temp;
	temp = error;
	// propagate the percent error to previous layers based on their relative weights to the output
	for (int i = (layers.size() - 1); i >= 0; i--) {
		cout << "Backpropagation on layer " << i << " starting" << endl;
		if (i == 0) temp = ((InputLayer *)layers[i])->backpropagate(temp, learningRate);
		else if (i == (layers.size() - 1)) temp = ((OutputLayer *)layers[i])->backpropagate(temp, learningRate);
		else if (i > 0 && (i < (layers.size() - 1))) temp = ((HiddenLayer *)layers[i])->backpropagate(temp, learningRate);
		cout << "Backpropagation on layer " << i << " finished" << endl;
	}
}

