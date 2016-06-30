/*
 * OutputLayer.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef OUTPUTLAYER_H_
#define OUTPUTLAYER_H_

#include "Layer.h"
#include <cuda.h>

__global__ void activateNeuron(double *input, Neuron nodes[], double *output);
__global__ void activateSynapse(double *input, Synapse connections[], double *output);
__global__ void sumInputFromSynapse(double *input, double *output);
__global__ void gradientDescent(double *error, double learningRate, Neuron nodes[], Neuron previous[], Synapse connections[]);
__global__ void sumWeightedError(double *error, Neuron nodes[], Synapse connections[], double *output);

class OutputLayer: public Layer {
public:
	bool debug;
	int currentLayerNeurons;
	int previousLayerNeurons;
	OutputLayer(int w, int d, bool db);
	virtual ~OutputLayer();
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate(vector<double> error, double learningRate, vector<Neuron> previousLayer);
};

#endif /* OUTPUTLAYER_H_ */
