/*
 * Network.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "NeuralNetwork.cuh"

long long NeuralNetwork::overhead = 0;

long long NeuralNetwork::computation = 0;

/**
 *
 * 	Initialize the Neural Network
 * 	Create and fill Layers
 *
 */
NeuralNetwork::NeuralNetwork(vector<int> size, double range, double rate, bool d) {
	// TODO Auto-generated constructor stub
	debug = d;
	learningRate = rate;
	errorPrime = vector<double>(size[size.size() - 1], 0);
	for (unsigned int i = 0; i < size.size(); i++) {
		if (i == 0) layers.push_back(new InputLayer(size[i], range, d));
		else if (i == (size.size() - 1)) layers.push_back(new OutputLayer(size[i], size[i - 1], d));
		else if (i > 0 && (i < (size.size() - 1))) layers.push_back(new HiddenLayer(size[i], size[i - 1], d));
	}
}

NeuralNetwork::~NeuralNetwork() {
	// TODO Auto-generated destructor stub
}


/**
 *
 * 	Feed an input vector into the Neural Network
 * 	Retrieve a classification vector
 *
 */
vector<double> NeuralNetwork::classify(vector<double> input) {
	if (input.size() == layers[0]->neurons.size()) return feedforward(input);
	else return vector<double>();
}


/**
 *
 * 	Feed the input vector forward into layers progressively
 * 	Output classification vector
 *
 */
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
	} return temp;
}


/**
 *
 * 	Train the Neural Network using Incremental updates
 *
 */
vector<vector<double> > NeuralNetwork::online(vector<double> input, vector<double> actual, double rate, bool print) {
	if (input.size() != layers[0]->neurons.size() ||
			actual.size() != layers[layers.size() - 1]->neurons.size()) {
		cout << "Illegal Argument at Network::train(vector<double> input, vector<double> actual) " << input.size() << " " << actual.size() << endl;
		return vector<vector<double> >();
	} else {
		vector<double> output, error;
		output = feedforward(input);
		learningRate = rate;
		double sum = 0;
		for (int i = 0; i < output.size(); i++) {
			errorPrime[i] = (output[i] - actual[i]) * layers[layers.size() - 1]->neurons[i].derivative;
			sum += (output[i] - actual[i]) * (output[i] - actual[i]);
		} sum /= (output.size() * 2);
		error.push_back(sum);
		if (print)cout << "error = " << sum << endl;

		vector<vector<double> > value;
		value.push_back(output);
		value.push_back(error);

		backpropagate();
		return value;
	}
}


/**
 *
 * 	Train the Neural Network using Batch updates
 *
 */
vector<vector<double> > NeuralNetwork::batch(vector<double> input, vector<double> actual, double rate, bool print, bool update) {
	if (input.size() != layers[0]->neurons.size() ||
			actual.size() != layers[layers.size() - 1]->neurons.size()) {
		cout << "Illegal Argument at Network::train(vector<double> input, vector<double> actual) " << input.size() << " " << actual.size() << endl;
		return vector<vector<double> >();
	} else {
		vector<double> output, error;
		vector<vector<double> > value;
		output = feedforward(input);

		if (update) {
			double sum = 0;
			for (int i = 0; i < output.size(); i++) {
				errorPrime[i] += (output[i] - actual[i]) * layers[layers.size() - 1]->neurons[i].derivative;
				sum += (output[i] - actual[i]) * (output[i] - actual[i]);
			} sum /= (output.size() * 2);
			error.push_back(sum);
			if (print)cout << "error = " << sum << endl;

			value.push_back(output);
			value.push_back(errorPrime);
			learningRate = rate;

			backpropagate();
		} return value;
	}
}


/**
 *
 * 	Backpropagate the error through the network
 * 	Update the weights and biases
 *
 */
vector<double> NeuralNetwork::backpropagate() {
	vector<double>  temp;
	temp = errorPrime;
	// propagate the percent error to previous layers based on their relative weights to the output
	for (int i = (layers.size() - 1); i >= 0; i--) {
		if (debug) cout << "Backpropagation on layer " << i << " starting" << endl;
		if (i == 0) temp = ((InputLayer *)layers[i])->backpropagate(temp, learningRate);
		else if (i == (int)(layers.size() - 1)) temp = ((OutputLayer *)layers[i])->backpropagate(temp, learningRate, layers[i - 1]->neurons);
		else if (i > 0 && (i < (int)(layers.size() - 1))) temp = ((HiddenLayer *)layers[i])->backpropagate(temp, learningRate, layers[i - 1]->neurons);
		if (debug) cout << "Backpropagation on layer " << i << " finished" << endl;
	} for (int i = 0; i < errorPrime.size(); i++) {
		errorPrime[i] = 0;
	} return temp;
}

struct tm *currentDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}


/**
 *
 * 	Save the current instance of the network
 * 	Output a file of weights and biases
 *
 */
void NeuralNetwork::toFile(int iteration, int numberTrainIterations, int repeatImages, double decay) {
	ostringstream fileName;
	fileName << "/stash/tlab/trabucco/ANN_Saves/" <<
			(currentDate()->tm_year + 1900) << "-" << (currentDate()->tm_mon + 1) << "-" << currentDate()->tm_mday <<
			"_GPU-ANN-Save-" << iteration << "_" <<
			numberTrainIterations <<
			"-iterations_" << repeatImages <<
			"-repeat_" << learningRate <<
			"-learning_" << decay << "-decay.csv";
	ofstream _file(fileName.str());

	for (int i = 1; i < layers.size(); i++) {
		if (i == (layers.size() - 1)) for (int j = 0; j < ((OutputLayer *)layers[i])->synapses.size(); j++) {
			if (j == (((OutputLayer *)layers[i])->synapses.size() - 1))_file << ((OutputLayer *)layers[i])->synapses[j].weight << endl;
			else _file << ((OutputLayer *)layers[i])->synapses[j].weight << ",";
		} else for (int j = 0; j < ((HiddenLayer *)layers[i])->synapses.size(); j++) {
			if (j == (((HiddenLayer *)layers[i])->synapses.size() - 1))_file << ((HiddenLayer *)layers[i])->synapses[j].weight << endl;
			else _file << ((HiddenLayer *)layers[i])->synapses[j].weight << ",";
		}
	}

	_file.close();
}

