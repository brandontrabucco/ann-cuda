/*
 * PassiveNeuron.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef PASSIVENEURON_H_
#define PASSIVENEURON_H_

#include "Neuron.h"

class PassiveNeuron: public Neuron {
public:
	PassiveNeuron();
	virtual ~PassiveNeuron();
	double get(double input, double scalar);
	double slope(double scalar);
};

#endif /* PASSIVENEURON_H_ */
