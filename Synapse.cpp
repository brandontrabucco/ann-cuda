/*
 * Synapse.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "Synapse.h"

Synapse::Synapse(int s) {
	// TODO Auto-generated constructor stub
	srand(time(0) + s);
	weight = (double)rand() / (double)RAND_MAX;
	bias =(double)rand() / (double)RAND_MAX;
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
