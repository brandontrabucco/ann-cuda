/*
 * PassiveNeuron.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef PASSIVENEURON_H_
#define PASSIVENEURON_H_

#include "Neuron.cuh"
#include <cuda.h>

class PassiveNeuron: public Neuron {
public:
	PassiveNeuron();
	virtual ~PassiveNeuron();
	__device__ __host__ double get(double input, double scalar);
};

#endif /* PASSIVENEURON_H_ */
