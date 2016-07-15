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
#include <math.h>
#include <numeric>
#include <string.h>
#include <fstream>
#include <sstream>

class NeuralNetwork {
private:
	bool debug;
	double learningRate;
	vector<Layer *> layers;
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate();
	vector<double> errorPrime;
public:
	NeuralNetwork(vector<int> size, double range, double rate, bool d);
	virtual ~NeuralNetwork();
	vector<double> classify(vector<double> input);
	vector<vector<double> > online(vector<double> input, vector<double> actual, double rate, bool print);
	vector<vector<double> > batch(vector<double> input, vector<double> actual, double rate, bool print, bool update);
	void toFile(int iteration, int trainingSet, int epoch, double decay);
};

#endif /* NETWORK_H_ */
