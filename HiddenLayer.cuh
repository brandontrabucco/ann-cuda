/*
 * HiddenLayer.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef HIDDENLAYER_H_
#define HIDDENLAYER_H_

#include "Layer.cuh"
#include "networkKernels.cuh"
#include <cuda.h>
#include <math.h>
#include <sys/time.h>

using namespace std;

class HiddenLayer: public Layer {
public:
	static int overhead, computation;
	bool debug;
	int currentLayerNeurons;
	int previousLayerNeurons;
	int kernelGridHeight;
	int kernelGridWidth;
	int kernelBlockHeight;
	int kernelBlockWidth;
	HiddenLayer(int w, int d, bool db);
	virtual ~HiddenLayer();
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate(vector<double> error, double learningRate, vector<Neuron> previousLayer);
};

#endif /* HIDDENLAYER_H_ */
