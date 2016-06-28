/*
 * Synapse.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef SYNAPSE_H_
#define SYNAPSE_H_

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <random>

using namespace std;

class Synapse {
private:
public:
	double weight;
	double bias;
	double input;
	double output;
	double weightedError;
	int index;
	Synapse(int s);
	virtual ~Synapse();
	double get(double i);
};

#endif /* SYNAPSE_H_ */
