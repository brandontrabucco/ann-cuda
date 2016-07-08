/*
 * HiddenLayer.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(int w, int d, bool db) {
	// TODO Auto-generated constructor stub
	debug = db;
	currentLayerNeurons = w;
	previousLayerNeurons = d;

	// add neurons and synapses to this layer
	for (int i = 0; i < w; i++) {
		Neuron n = Neuron();
		n.index = i;
		neurons.push_back(n);
		if (debug) cout << "Neuron created " << i << endl;
	} for (int i = 0; i < (w * d); i++) {
		Synapse s = Synapse();
		s.index = i;
		synapses.push_back(s);
		if (debug) cout << "Synapse created " << i << endl;
	}
}

HiddenLayer::~HiddenLayer() {
	// TODO Auto-generated destructor stub
}

vector<double> HiddenLayer::feedforward(vector<double> input) {
	vector<double> temp, sum, output;	// variables to store data for math operations
		for (int i = 0; i < currentLayerNeurons; i++) {	// iterate through each synapse
			for (int j = 0; j < previousLayerNeurons; j++) {
				// calculate a synapse input for each connection
				temp.push_back(synapses[(i * previousLayerNeurons) + j].get(input[j]));
				//if (j == 0) cout << synapses[(i * previousLayerNeurons) + j].weight << endl;
			}
		} for (int i = 0; i < currentLayerNeurons; i++) {	// iterate through each synapse for input
			sum.push_back(0);
			for (int j = 0; j < previousLayerNeurons; j++) {
				sum[i] += temp[(i * previousLayerNeurons) + j];
			} //sum[i] /= (temp.size());
			output.push_back(neurons[i].get(sum[i]));
			if (debug) cout << "Neuron " << neurons[i].index << " activating by " << output[i] << endl;
		} return output;
}

vector<double> HiddenLayer::backpropagate(vector<double> error, double learningRate, vector<Neuron> previousLayer) {
	// iterate through each synapse connected to the previous layer
	vector<double> eta, sum;
	for (int i = 0; i < currentLayerNeurons; i++) {
		eta.push_back(error[i] * neurons[i].derivative);
		for (int j = 0; j < previousLayerNeurons; j++) {
			if (i == 0) sum.push_back(0);
			// update the weight and bias variables (need to take the weighted error in proportion to the sum of weights to a neuron)
			synapses[(i * previousLayerNeurons) + j].weight -= learningRate * eta[i] * previousLayer[j].activation;
			//synapses[(i * previousLayerNeurons) + j].bias -= learningRate * eta[i];
			sum[j] += ((eta[i] * synapses[(i * previousLayerNeurons) + j].weight) + synapses[(i * previousLayerNeurons) + j].bias);
			//sum[j] /= previousLayerNeurons;
		}
	} return sum;
}

