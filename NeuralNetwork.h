/*
 * Network.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_
#define DEBUG true

#include "Layer.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

class NeuralNetwork {
private:
	bool debug;
	double learningRate;
	vector<Layer *> layers;
	vector<double> feedforward(vector<double> input);
	void backpropagate(vector<double> error);
public:
	NeuralNetwork(vector<int> size, double range, double rate, bool d);
	virtual ~NeuralNetwork();
	void classify(double input);
	void classify(vector<double> input);
	void train(double input, double actual);
	void train(vector<double> input, vector<double> actual);
};

#endif /* NETWORK_H_ */
