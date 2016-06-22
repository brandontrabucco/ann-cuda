/*
 * Synapse.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef SYNAPSE_H_
#define SYNAPSE_H_

#include <stdlib.h>

class Synapse {
public:
	double weight;
	double bias;
	double input;
	double output;
	double weightedError;
	Synapse();
	virtual ~Synapse();
	double get(double i);
};

#endif /* SYNAPSE_H_ */
