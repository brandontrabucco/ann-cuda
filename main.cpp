//============================================================================
// Name        : main.cpp
// Author      : Brandon Trabucco
// Version     : 1.0.2
// Copyright   : This project is licensed under the GNU General Public License
// Description : This project is a test implementation of a Neural Network
//============================================================================

#include "Network.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
	vector<int> size;
	size.push_back(2);	// layer 0
	size.push_back(1);	// layer 3

	// The base class for our neural network
	Network network = Network(size, 100, 1);

	// teach the network
	network.train(100.0, 1);
	network.train(45.0, 0);
	network.train(95.0, 1);
	network.train(40.0, 0);
	network.train(90.0, 1);
	network.train(35.0, 0);
	network.train(85.0, 1);
	network.train(30.0, 0);
	network.train(80.0, 1);
	network.train(25.0, 0);
	network.train(75.0, 1);
	network.train(20.0, 0);
	network.train(70.0, 1);
	network.train(15.0, 0);
	network.train(65.0, 1);
	network.train(10.0, 0);
	network.train(60.0, 1);
	network.train(05.0, 0);
	network.train(55.0, 1);
	network.train(00.0, 0);

	// test the network for a success
	network.classify(21.3);
	network.classify(89.7);

	return 0;
}
