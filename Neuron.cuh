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
	__device__ __host__ Neuron();
	__device__ __host__ virtual ~Neuron();
	__device__ __host__ virtual double get(double input);
	__device__ __host__ virtual double slope(double input);
};

#endif /* NEURON_H_ */
