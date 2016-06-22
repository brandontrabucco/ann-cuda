/*
 * Neuron.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "Neuron.h"

Neuron::Neuron() {
	// TODO Auto-generated constructor stub

}

Neuron::~Neuron() {
	// TODO Auto-generated destructor stub
}

double Neuron::get(double input) {
	activation = 1 / (1 + exp(-input));
	return activation;
}

double Neuron::slope(double input) {
	derivative = get(input) * (1 - get(input));
	return derivative;
}

