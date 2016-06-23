/*
 * Network.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "Layer.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

class Network {
private:
	double learningRate;
	vector<Layer *> layers;
	vector<double> feedforward(vector<double> input);
	void backpropagate(vector<double> error);
public:
	Network(vector<int> size, double range, double rate);
	virtual ~Network();
	void classify(double input);
	void classify(vector<double> input);
	void train(double input, double actual);
	void train(vector<double> input, vector<double> actual);
};

#endif /* NETWORK_H_ */
