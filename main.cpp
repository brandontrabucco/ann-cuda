//============================================================================
// Name        : main.cpp
// Author      : Brandon Trabucco
// Version     : 1.0.4
// Copyright   : This project is licensed under the GNU General Public License
// Description : This project is a test implementation of a Neural Network
//============================================================================

#include "NeuralNetwork.h"
#include "ImageLoader.h"
#include "OutputTarget.h"
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <numeric>

using namespace std;

double getMSec() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

struct tm *getDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}

int main(int argc, char *argv[]) {
	cout << "Program initializing" << endl;
	if (argc != 4) {
		cout << argv[0] << " <training iterations> <learning rate> <decay rate>" << endl;
		return -1;
	}

	vector<int> size;
	vector<vector<double> > images;
	vector<double> labels;
	int numberImages = 0;
	int imageSize = 0;
	int numberLabels = 0;
	int numberTrainIterations = atoi(argv[1]);
	int numberTestIterations = atoi(argv[1]) / 10;
	int updatePoints = 100;
	double learningRate = atof(argv[2]), decay = atof(argv[3]);
	long long startTime, endTime;

	// open file streams with unique names
	ostringstream errorDataFileName;
	errorDataFileName << "/u/trabucco/Desktop/MNIST_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << getDate()->tm_mday <<
			"_cuda-error-data_" <<
			numberTrainIterations <<
			"-" << learningRate <<
			"-" << decay << ".csv";
	ostringstream timingDataFileName;
	timingDataFileName << "/u/trabucco/Desktop/MNIST_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << getDate()->tm_mday <<
			"_cuda-timing-data_" <<
			numberTrainIterations <<
			"-" << learningRate <<
			"-" << decay << ".csv";;
	ostringstream accuracyDataFileName;
	accuracyDataFileName << "/u/trabucco/Desktop/MNIST_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << getDate()->tm_mday <<
			"_cuda-accuracy-data_" <<
			numberTrainIterations <<
			"-" << learningRate <<
			"-" << decay << ".csv";

	ofstream errorData(errorDataFileName.str());
	if (!errorData.is_open()) return -1;
	ofstream timingData(timingDataFileName.str());
	if (!timingData.is_open()) return -1;
	ofstream accuracyData(accuracyDataFileName.str());
	if (!accuracyData.is_open()) return -1;

	accuracyData << numberTrainIterations << "," << learningRate << "," << decay;

	// load the images and get their sizes
	startTime = getMSec();
	images = ImageLoader::readMnistImages("/u/trabucco/Desktop/MNIST_Bytes/train-images.idx3-ubyte", numberImages, imageSize);
	labels = ImageLoader::readMnistLabels("/u/trabucco/Desktop/MNIST_Bytes/train-labels.idx1-ubyte", numberLabels);
	endTime = getMSec();
	cout << "Training images and labels loaded in " << (endTime - startTime) << " msecs" << endl;

	size.push_back(imageSize);	// layer 0
	size.push_back(imageSize / 2);	// layer 1
	size.push_back(imageSize / 4);	// layer 2
	size.push_back(imageSize / 8);	// layer 3
	size.push_back(imageSize / 16);	// layer 4
	size.push_back(imageSize / 32);	// layer 5
	size.push_back(10);	// layer 6

	// the base class for our neural network
	NeuralNetwork network = NeuralNetwork(size, 1.0, learningRate, false);

	// iterate the network through each image and pixel
	for (int i = 0; i < numberTrainIterations; i++) {
		startTime = getMSec();
		vector<double> error = network.train(images[i], OutputTarget::getTargetOutput(labels[i]), learningRate, !(i % (numberTrainIterations / 100)));
		errorData << i << "," << (accumulate( error.begin(), error.end(), 0.0)/error.size())  << endl;
		endTime = getMSec();
		if (!(i % (numberTrainIterations / updatePoints))) {
			errorData << i;
			errorData << "," << ((double)accumulate(error.begin(), error.end(), 0.0)) / ((double)error.size());
			errorData << endl;
			timingData << i << "," << (endTime - startTime) << endl;
			cout << "Iteration " << i << " " << (endTime - startTime) << "msecs, ETA " << (((double)(endTime - startTime)) * (numberTrainIterations - (double)i) / 1000.0 / 60.0) << "min" << endl;
			learningRate *= decay;
		}
	}

	// load test images
	images = ImageLoader::readMnistImages("/u/trabucco/Desktop/MNIST_Bytes/t10k-images.idx3-ubyte", numberImages, imageSize);
	labels = ImageLoader::readMnistLabels("/u/trabucco/Desktop/MNIST_Bytes/t10k-labels.idx1-ubyte", numberLabels);

	// iterate through each test image
	int c = 0;
	for (int i = 0; i < numberTestIterations; i++) {
		startTime = getMSec();
		vector<double> temp = network.classify(images[i]);
		endTime = getMSec();

		if (!(i % (numberTestIterations / updatePoints))) {
			cout << "Iteration " << i << " " << (endTime - startTime) << "msecs, ETA " << (((double)(endTime - startTime)) * (numberTestIterations - (double)i) / 1000.0 / 60.0) << "min" << endl;
			cout << "Classification " << OutputTarget::getTargetFromOutput(temp) << " : actual " << labels[i] << endl;
			if (OutputTarget::getTargetFromOutput(temp) == labels[i]) {
				c += 1;
			}
			for (int j = 0; j < temp.size(); j++) {
				cout << "output[" << j << "] : " << temp[j] << endl;
			}
		}
	}

	// report on how accurate the network was when testing
	accuracyData << "," << learningRate << "," << (((double)c) / ((double)numberTestIterations)) << endl;

	cout << "Program finished" << endl;

	errorData.close();
	timingData.close();
	accuracyData.close();

	return 0;
}
