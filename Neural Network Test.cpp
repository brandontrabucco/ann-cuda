//============================================================================
// Name        : Neural.cpp
// Author      : Brandon Trabucco
// Version     : 1.0.0
// Copyright   : This project is licensed under the GNU General Public License
// Description : This project is a test implementation of a Neural Network
//============================================================================

///
/// Compiler directives
///

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PASSIVE_NEURONS 16
#define LAYER_1_NEURONS 16
#define LAYER_2_NEURONS 2
#define LEARNING_DECAY_ALPHA 1
#define LEARNING_DECAY_BETA 1


// Set namespace
using namespace std;

// Declare functions
double activation_function(double input);
double compute_weighted_input(int layer, int sender, int receiver, double **_data, double ***_weights);
void update_weight(int layer, int sender, int receiver, double **_data, double ***_weights, double ***_learning, double output, double actual);
double classify_data(double **_data, double ***_weights);
double back_propogate(double **_data, double **_weights, double **_answers);

///
/// Entry point
///

int main() {
	// Variables
	double **data;
	double ***weights;
	double ***learning;

	// Allocate memory for each variable
	data = (double **)malloc(sizeof(double) * (LAYER_1_NEURONS + LAYER_2_NEURONS));	// One double slot per neuron
	weights = (double ***)malloc(sizeof(double) * (LAYER_1_NEURONS * LAYER_2_NEURONS));	// One double slot per connection between the active neurons
	learning = (double ***)malloc(sizeof(double) * (LAYER_1_NEURONS * LAYER_2_NEURONS));	// One double slot per connection between the active neurons

	cout << "This is some random text!!!" << endl;

	// Free the allocated memory
	free(data);
	free(weights);
	free(learning);

	return 0;
}

///
/// Function definitions
///

// The non linear activation function of a neuron
double activation_function(double input) {
	return tanh(input);
}

// Calculate weighted input
double compute_weighted_input(int layer, int sender, int receiver, double **_data, double ***_weights) {
	return _weights[layer][sender][receiver] * _data[layer][sender];
}

// Update the weights of a connection based on the error present using gradient descent
void update_weight(int layer, int sender, int receiver, double **_data, double ***_weights, double ***_learning, double output, double actual) {
	// Squared error function is used E(o) = 1/2*(target - output)^2
	double gradient = (output - actual);

	// Update the weight
	_weights[layer][sender][receiver] += gradient * _learning[layer][sender][receiver];

	// Update the learning rate
	_learning[layer][sender][receiver] *= LEARNING_DECAY_ALPHA * (1 - _data[layer][sender]);
}

// Pass a data set through the network and calculate activation values
double classify_data(double **_data, double ***_weights) {
	// Per neuron in first layer
	for (int i = 0; i < LAYER_1_NEURONS; i++) {
		_data[0][i] = activation_function(_data[0][i]);
	}

	// Per neuron in second layer
	for (int i = 0; i < LAYER_2_NEURONS; i++) {
		double input = 0;

		// Sum up the weighted input of previous neurons
		for (int j = 0; j < LAYER_1_NEURONS; j++) {
			input += compute_weighted_input(0, j, i, _data, _weights);
		}

		// Compute the activation of current
		_data[1][i] = pow(activation_function(_data[1][i]), 2);
	}

	return sqrt(pow(_data[1][0], 2) + pow(_data[1][1], 2));
}

// Train the neural network
void back_propogate(double **_data, double ***_weights, double ***_learning, double **_answers) {
	classify_data(_data, _weights);

	// Per neuron in second layer
	for (int i = 0; i < LAYER_2_NEURONS; i++) {
		// Iterate for every connection
		for (int j = 0; j < LAYER_1_NEURONS; j++) {
			update_weight(1, j, i, _data, _weights, _learning, _data[0][j], _answers[0][j]);
		}
	}
}

