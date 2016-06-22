/*
 * Synapse.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "Synapse.h"

Synapse::Synapse() {
	// TODO Auto-generated constructor stub
	weight = (double)rand() / (double)RAND_MAX;
	bias =(double)rand() / (double)RAND_MAX;
}

Synapse::~Synapse() {
	// TODO Auto-generated destructor stub
}

double Synapse::get(double i) {
	input = i;
	output = (weight * input) + bias;
	return output;
}
