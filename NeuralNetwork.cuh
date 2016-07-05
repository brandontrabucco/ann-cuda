/*
 * Network.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_
#define DEBUG true

#include "Layer.cuh"
#include "InputLayer.cuh"
#include "HiddenLayer.cuh"
#include "OutputLayer.cuh"
#include "networkKernels.cuh"
#include <math.h>
#include <numeric>
#include <cuda.h>

class NeuralNetwork {
private:
	bool debug;
	double learningRate;
	vector<Layer *> layers;
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate(vector<double> error);
public:
	NeuralNetwork(vector<int> size, double range, double rate, bool d);
	virtual ~NeuralNetwork();
	vector<double> classify(double input);
	vector<double> classify(vector<double> input);
	vector<vector<double> > train(double input, double actual, double rate, bool print);
	vector<vector<double> > train(vector<double> input, vector<double> actual, double rate, bool print);
};

#endif /* NETWORK_H_ */