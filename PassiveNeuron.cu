/*
 * PassiveNeuron.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "PassiveNeuron.cuh"

PassiveNeuron::PassiveNeuron() : Neuron() {
	// TODO Auto-generated constructor stub

}

PassiveNeuron::~PassiveNeuron() {
	// TODO Auto-generated destructor stub
}

__device__ __host__ double PassiveNeuron::get(double input, double scalar) {
	activation = input * scalar;
	derivative = scalar;

	return 0;
}
