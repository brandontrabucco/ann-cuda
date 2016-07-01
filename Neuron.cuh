/*
 * Neuron.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <math.h>
#include <cuda.h>

class Neuron {
public:
	double activation;
	double derivative;
	int index;
	Neuron();
	virtual ~Neuron();
	__device__ virtual double get(double input);
};

#endif /* NEURON_H_ */
