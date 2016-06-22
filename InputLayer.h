/*
 * InputLayer.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef INPUTLAYER_H_
#define INPUTLAYER_H_

#include "Layer.h"
#include "PassiveNeuron.h"

class InputLayer: public Layer {
public:
	int width;
	int depth;
	double scalar;
	InputLayer(int width, int depth, double range);
	virtual ~InputLayer();
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate(vector<double> error, double learningRate);
};

#endif /* INPUTLAYER_H_ */
