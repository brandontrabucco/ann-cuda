/*
 * Neuron.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <math.h>

class Neuron {
public:
	double activation;
	double derivative;
	int index;
	Neuron();
	virtual ~Neuron();
	virtual double get(double input);
	virtual double slope(double input);
};

#endif /* NEURON_H_ */
