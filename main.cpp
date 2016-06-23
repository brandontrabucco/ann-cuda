//============================================================================
// Name        : main.cpp
// Author      : Brandon Trabucco
// Version     : 1.0.3
// Copyright   : This project is licensed under the GNU General Public License
// Description : This project is a test implementation of a Neural Network
//============================================================================

#include "NeuralNetwork.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
	vector<int> size;

	size.push_back(1);	// layer 0
	size.push_back(3);	// layer 1
	size.push_back(2);	// layer 2

	// the base class for our neural network
	NeuralNetwork network = NeuralNetwork(size, 10.0, .5, false);

	// teach the network to differentiate between high and low
	for (double i = 10; i >= 0; i -= .00001) {
		network.train(i, ((i > 5) ? 1 : 0));
		cout << "For " << i << endl;
		network.train((10 - i), (((10 - i) > 5) ? 1 : 0));
		cout << "For " << (10 - i) << endl;
	}

	//test the network for a success
	cout << endl;
	network.classify(0);
	network.classify(10);

	return 0;
}
