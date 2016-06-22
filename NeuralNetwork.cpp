#include "NeuralNetwork.h"
#include <armadillo>
#include <vector>
#include <math.h>
#include <iostream>

using namespace std;
using namespace arma;

NeuralNetwork::NeuralNetwork(vector<int> s) {
	sizes = s;
	layers = s.size();
	for (int i = 1; i < layers - 1; i++) {
		biases[i] = mat();
		biases[i].randu(sizes[i] * sizes[i + 1]);
		weights[i] = mat();
		weights[i].randu(sizes[i] * sizes[i + 1]);
	}
}


NeuralNetwork::~NeuralNetwork() {
}


void NeuralNetwork::feedForward(mat &a) {
	for (int i = 1; i < layers - 1; i++) {
		a = dot(weights[i], a) + biases[i];
		mat temp;
		for (int j = 0; j < a.n_elem; j++) {
			temp << sigmoid(a.at(j)) << endr;
		}
		a = temp;
	}
}


void NeuralNetwork::gradientDescent(mat trainingData, int epochs, double learningRate) {
	for (int i = 0; i < epochs; i++) {
		mat differentials = backPropogate(trainingData);
	}
}


mat NeuralNetwork::backPropogate(mat trainingData) {
	mat *activations;
	mat *inputs;
	mat *prime;
	for (int i = 1; i < layers - 1; i++) {
		inputs[i] = dot(weights[i], trainingData.col(0)) + biases[i];
		for (int j = 0; j < trainingData.n_rows; j++) {
			activations[i] << sigmoid(inputs[i].at(j)) << endr;
			prime[i] << sigmoidPrime(inputs[i].at(j)) << endr;
		}
	}
	mat delta;
	for (int i = 1; i < layers - 1; i++) {

	}
}


double NeuralNetwork::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}


double NeuralNetwork::sigmoidPrime(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}
