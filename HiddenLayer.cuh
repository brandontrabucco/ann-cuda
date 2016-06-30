/*
 * HiddenLayer.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef HIDDENLAYER_H_
#define HIDDENLAYER_H_

#include "Layer.h"
#include <cuda.h>

using namespace std;

__global__ void activateNeuron(double *input, Neuron nodes[], double *output);
__global__ void activateSynapse(double *input, Synapse connections[], double *output);
__global__ void sumInputFromSynapse(double *input, double *output);
__global__ void gradientDescent(double *error, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[]);
__global__ void sumWeightedError(double *error, Neuron nodes[], Synapse connections[], double *output);

class HiddenLayer: public Layer {
public:
	bool debug;
	int currentLayerNeurons;
	int previousLayerNeurons;
	HiddenLayer(int w, int d, bool db);
	virtual ~HiddenLayer();
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate(vector<double> error, double learningRate, vector<Neuron> previousLayer);
};

#endif /* HIDDENLAYER_H_ */
