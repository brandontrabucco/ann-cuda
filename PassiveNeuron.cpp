/*
 * PassiveNeuron.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "PassiveNeuron.h"

PassiveNeuron::PassiveNeuron() : Neuron() {
	// TODO Auto-generated constructor stub

}

PassiveNeuron::~PassiveNeuron() {
	// TODO Auto-generated destructor stub
}

double PassiveNeuron::get(double input, double scalar) {
	return input * scalar;
}

double PassiveNeuron::slope(double scalar) {
	return scalar;
}
