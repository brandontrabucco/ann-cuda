/*
 * InputLayer.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef INPUTLAYER_H_
#define INPUTLAYER_H_

#include "Layer.cuh"
#include "PassiveNeuron.cuh"
#include "networkKernels.cuh"
#include "NeuralNetwork.cuh"
#include <cuda.h>

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
