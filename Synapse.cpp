/*
 * Synapse.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "Synapse.cuh"

normal_distribution<double> Synapse::d(0, 1);

default_random_engine Synapse::g;

Synapse::Synapse() {
	// TODO Auto-generated constructor stub
	weight = d(g);
	bias = 0;
	weightedError = 0;
}

Synapse::~Synapse() {
	// TODO Auto-generated destructor stub
}

double Synapse::get(double i) {
	input = i;
	output = (weight * input) + bias;
	return output;
}
