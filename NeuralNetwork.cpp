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
		if (i == 0) layers.push_back(new InputLayer(size[i], range, d));
		else if (i == (size.size() - 1)) layers.push_back(new OutputLayer(size[i], size[i - 1], d));
		else if (i > 0 && (i < (size.size() - 1))) layers.push_back(new HiddenLayer(size[i], size[i - 1], d));
	}
}

NeuralNetwork::~NeuralNetwork() {
	// TODO Auto-generated destructor stub
}

vector<double> NeuralNetwork::classify(double input) {
	vector<double> temp;
	for (unsigned int i = 0; i < layers[0]->neurons.size(); i++) temp.push_back(input);
	return feedforward(temp);
}

vector<double> NeuralNetwork::classify(vector<double> input) {
	if (input.size() == layers[0]->neurons.size()) return feedforward(input);
	else return vector<double>();
}

vector<double> NeuralNetwork::feedforward(vector<double> input) {
	vector<double> temp;
	temp = input;
	for (unsigned int i = 0; i < layers.size(); i++) {
		// feed the input through each layer of network
		if (debug) cout << endl << "Layer " << i << " propagating" << endl << endl;
		if (i == 0) {
			temp = ((InputLayer *)layers[i])->feedforward(temp);
		} else if (i == (layers.size() - 1)) {
			temp = ((OutputLayer *)layers[i])->feedforward(temp);
		} else if (i > 0 && (i < (layers.size() - 1))) {
			temp = ((HiddenLayer *)layers[i])->feedforward(temp);
		} if (debug) cout << endl << "Layer " << i << " finished" << endl << endl;
	} for (unsigned int i = 0; i < layers[layers.size() - 1]->neurons.size(); i++) {
		// show the output of the last layer
		//cout << "Output Neuron " << i << " : " << temp[i] << endl;
	}
	return temp;
}

vector<vector<double> > NeuralNetwork::train(double input, double actual, double rate, bool print) {
	vector<double> temp, output, error;
	for (unsigned int i = 0; i < layers[0]->neurons.size(); i++) temp.push_back(input);
	output = feedforward(temp);
	learningRate = rate;

	// get error with respect to each of the output nodes
	for (unsigned int i = 0; i < output.size(); i++) {
		error.push_back((output[i] - actual));
		if (print) cout << "error[" << i << "]" << error[i] << endl;
	}
	if (print) cout << endl;
	backpropagate(error);

	vector<vector<double> > value;
	value.push_back(output);
	value.push_back(error);

	return value;
}

vector<vector<double> > NeuralNetwork::train(vector<double> input, vector<double> actual, double rate, bool print) {
	if (input.size() != layers[0]->neurons.size() ||
			actual.size() != layers[layers.size() - 1]->neurons.size()) {
		cout << "Illegal Argument at Network::train(vector<double> input, vector<double> actual) " << input.size() << " " << actual.size() << endl;
		return vector<vector<double> >();
	} else {
		vector<double> output, error;
		output = feedforward(input);
		learningRate = rate;
		for (unsigned int i = 0; i < output.size(); i++) {
			error.push_back((output[i] - actual[i]));
			if (print) cout << "error[" << i << "]" << error[i] << endl;
		}
		if (print) cout << endl;
		backpropagate(error);

		vector<vector<double> > value;
		value.push_back(output);
		value.push_back(error);

		return value;
	}
}

vector<double> NeuralNetwork::backpropagate(vector<double> error) {
	vector<double>  temp;
	temp = error;
	// propagate the percent error to previous layers based on their relative weights to the output
	for (int i = (layers.size() - 1); i >= 0; i--) {
		if (debug) cout << "Backpropagation on layer " << i << " starting" << endl;
		if (i == 0) temp = ((InputLayer *)layers[i])->backpropagate(temp, learningRate);
		else if (i == (int)(layers.size() - 1)) temp = ((OutputLayer *)layers[i])->backpropagate(temp, learningRate, layers[i - 1]->neurons);
		else if (i > 0 && (i < (int)(layers.size() - 1))) temp = ((HiddenLayer *)layers[i])->backpropagate(temp, learningRate, layers[i - 1]->neurons);
		if (debug) cout << "Backpropagation on layer " << i << " finished" << endl;
	} return temp;
}

struct tm *currentDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}

void NeuralNetwork::toFile(int iteration, int numberTrainIterations, double decay) {
	ostringstream fileName;
	fileName << "/u/trabucco/Desktop/ANN_Saves/" <<
			(currentDate()->tm_year + 1900) << "-" << (currentDate()->tm_mon + 1) << "-" << currentDate()->tm_mday <<
			"_CPU-ANN-Save-Iteration-" << iteration << "_" <<
			numberTrainIterations <<
			"-" << learningRate <<
			"-" << decay << ".ann";
	ofstream _file(fileName.str());

	for (int i = 1; i < layers.size(); i++) {
		if (i == (layers.size() - 1)) for (int j = 0; j < ((OutputLayer *)layers[i])->synapses.size(); j++) {
			if (j == (((OutputLayer *)layers[i])->synapses.size() - 1))_file << ((OutputLayer *)layers[i])->synapses[j]->weight << endl;
			else _file << ((OutputLayer *)layers[i])->synapses[j]->weight << ",";
		} else for (int j = 0; j < ((HiddenLayer *)layers[i])->synapses.size(); j++) {
			if (j == (((HiddenLayer *)layers[i])->synapses.size() - 1))_file << ((HiddenLayer *)layers[i])->synapses[j]->weight << endl;
			else _file << ((HiddenLayer *)layers[i])->synapses[j]->weight << ",";
		}
	}

	_file.close();
}

