#pragma once
#include <armadillo>

using namespace std;
using namespace arma;
class NeuralNetwork
{
private:
	int layers;
	vector<int> sizes;
	mat *biases;
	mat *weights;
public:
	NeuralNetwork(vector<int> s);
	~NeuralNetwork();
	void feedForward(mat &a);
	void gradientDescent(mat trainingData, int epochs, double learningRate);
	mat backPropogate(mat);
	double sigmoid(double x);
	double sigmoidPrime(double x);
};

