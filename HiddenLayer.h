/*
 * HiddenLayer.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef HIDDENLAYER_H_
#define HIDDENLAYER_H_

#include "Layer.h"
#include "Synapse.h"

using namespace std;

class HiddenLayer: public Layer {
public:
	int width;
	int depth;
	HiddenLayer(int w, int d);
	virtual ~HiddenLayer();
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate(vector<double> error, double learningRate);
};

#endif /* HIDDENLAYER_H_ */
