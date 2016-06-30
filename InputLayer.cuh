/*
 * InputLayer.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef INPUTLAYER_H_
#define INPUTLAYER_H_

#include "Layer.h"
#include "PassiveNeuron.cuh"
#include <cuda.h>

__global__ void activateNeuron(double *input, PassiveNeuron nodes[], double scalar, double *output);

class InputLayer: public Layer {
private:
public:
	bool debug;
	int currentLayerNeurons;
	double scalar;
	InputLayer(int w, double range, bool db);
	virtual ~InputLayer();
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate(vector<double> errorPrime, double learningRate);
};

#endif /* INPUTLAYER_H_ */
