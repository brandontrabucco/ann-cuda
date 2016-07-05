/*
 * Layer.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "Neuron.cuh"
#include "Synapse.cuh"
#include <vector>
#include <iostream>

using namespace std;

class Layer {
public:
	vector<Neuron> neurons;
	vector<Synapse> synapses;
	Layer();
	~Layer();
	vector<double> feedforward(vector<double> input);
	vector<double> backpropagate(vector<double> error);
};

#endif /* LAYER_H_ */
