/*
 * Network.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(vector<int> size, double range, double rate, bool d) {
	// TODO Auto-generated constructor stub
	debug = d;
	learningRate = rate;
	for (unsigned int i = 0; i < size.size(); i++) {
		if (i == 0) layers.push_back(new InputLayer(size[i], size[i + 1], range, d));
		else if (i == (size.size() - 1)) layers.push_back(new OutputLayer(size[i], d));
		else if (i > 0 && (i < (size.size() - 1))) layers.push_back(new HiddenLayer(size[i], size[i + 1], d));
	}
}

NeuralNetwork::~NeuralNetwork() {
	// TODO Auto-generated destructor stub
}

void NeuralNetwork::classify(double input) {
	vector<double> temp;
	for (unsigned int i = 0; i < layers[0]->neurons.size(); i++) temp.push_back(input);
	feedforward(temp);
}

void NeuralNetwork::classify(vector<double> input) {
	if (input.size() == layers[0]->neurons.size()) feedforward(input);
	else cout << "Illegal Argument at Network::classify(vector<double> input)" << endl;
}

vector<double> NeuralNetwork::feedforward(vector<double> input) {
	vector<double> temp;
	temp = input;
	for (unsigned int i = 0; i < layers.size(); i++) {
		// feed the input through each layer of network
		if (debug) cout << endl << "Layer " << i << " propagating" << endl << endl;
		if (i == 0) temp = ((InputLayer *)layers[i])->feedforward(temp);
		else if (i == (layers.size() - 1)) temp = ((OutputLayer *)layers[i])->feedforward(temp);
		else if (i > 0 && (i < (layers.size() - 1))) temp = ((HiddenLayer *)layers[i])->feedforward(temp);
		if (debug) cout << endl << "Layer " << i << " finished" << endl << endl;
	} for (unsigned int i = 0; i < layers[layers.size() - 1]->neurons.size(); i++) {
		// show the output of the last layer
		cout << "Output of Neuron " << i << " : " << temp[i] << endl;
	}
	return temp;
}

void NeuralNetwork::train(double input, double actual) {
	vector<double> temp, output, error;
	for (unsigned int i = 0; i < layers[0]->neurons.size(); i++) temp.push_back(input);
	output = feedforward(temp);

	// get error with respect to each of the output nodes
	for (unsigned int i = 0; i < output.size(); i++) {
		error.push_back(output[i] - actual);
		if (debug) cout << "Output Neuron " << i << " error " << error[i] << endl;
	}
	backpropagate(error);
}

void NeuralNetwork::train(vector<double> input, vector<double> actual) {
	if (input.size() != layers[0]->neurons.size() ||
			actual.size() != layers[layers.size() - 1]->neurons.size()) {
		cout << "Illegal Argument at Network::train(vector<double> input, vector<double> actual)" << endl;
		return;
	} else {
		vector<double> output, errorPrime;
		output = feedforward(input);

		// get error with respect to each of the output nodes
		for (unsigned int i = 0; i < output.size(); i++) {
			errorPrime.push_back((output[i] - actual[i]) * output[i] * (1 - output[i]));
			if (debug) cout << "Output Neuron " << i << " errorPrime " << errorPrime[i] << endl;
		}
		backpropagate(errorPrime);
	}
}

void NeuralNetwork::backpropagate(vector<double> errorPrime) {
	vector<double>  temp;
	temp = errorPrime;
	// propagate the percent error to previous layers based on their relative weights to the output
	for (int i = (layers.size() - 1); i >= 0; i--) {
		if (debug) cout << "Backpropagation on layer " << i << " starting" << endl;
		if (i == 0) temp = ((InputLayer *)layers[i])->backpropagate(temp, learningRate);
		else if (i == (layers.size() - 1)) temp = ((OutputLayer *)layers[i])->backpropagate(temp, learningRate);
		else if (i > 0 && (i < (layers.size() - 1))) temp = ((HiddenLayer *)layers[i])->backpropagate(temp, learningRate);
		if (debug) cout << "Backpropagation on layer " << i << " finished" << endl;
	}
}

