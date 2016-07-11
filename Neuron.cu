/*
 * Neuron.cu
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "Neuron.cuh"

Neuron::Neuron() {
	// TODO Auto-generated constructor stub

}

Neuron::~Neuron() {
	// TODO Auto-generated destructor stub
}

__device__ __host__ double Neuron::get(double input) {
	activation = 1 / (1 + exp(-input));
	derivative = (1 / (1 + exp(-input))) * (1 - (1 / (1 + exp(-input))));
	return activation;
}
