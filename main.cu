//============================================================================
// Name        : main.cpp
// Author      : Brandon Trabucco
// Version     : 1.0.4
// Copyright   : This project is licensed under the GNU General Public License
// Description : This project is a test implementation of a Neural Network accelerated with CUDA 7.5
//============================================================================

#include "NeuralNetwork.cuh"
#include "ImageLoader.h"
#include "OutputTarget.h"
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <numeric>
#include <cuda.h>

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
	int numberTestIterations = 100;
	int updatePoints = 100;
	double learningRate = atof(argv[2]), decay = atof(argv[3]);
	long long startTime, endTime, minTime;

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

	// load the images and get their sizes
	startTime = getMSec();
	images = ImageLoader::readMnistImages("/u/trabucco/Desktop/MNIST_Bytes/train-images.idx3-ubyte", numberImages, imageSize);
	labels = ImageLoader::readMnistLabels("/u/trabucco/Desktop/MNIST_Bytes/train-labels.idx1-ubyte", numberLabels);
	endTime = getMSec();
	cout << "Training images and labels loaded in " << (endTime - startTime) << " msecs" << endl;

	size.push_back(imageSize);
	size.push_back(imageSize);
	size.push_back(imageSize);
	size.push_back(imageSize);
	size.push_back(imageSize);
	size.push_back(imageSize);
	size.push_back(imageSize);
	size.push_back(10);					// layer 3

	// the base class for our neural network
	NeuralNetwork network = NeuralNetwork(size, 1.0, learningRate, false);

	// iterate the network through each image and pixel
	int c = 0;
	for (int i = 0; i < numberTrainIterations; i++) {
		startTime = getMSec();
		vector<vector<double> > trainingData = network.train(images[i], OutputTarget::getTargetOutput(labels[i]), learningRate, !(i % (numberTrainIterations / updatePoints)));
		endTime = getMSec();
		if (OutputTarget::getTargetFromOutput(trainingData[0]) == labels[i]) {
			c += 1;
		} if (!(i % (numberTrainIterations / updatePoints))) {
			double errorSum = 0;

			for (int j = 0; j < trainingData[1].size(); j++) {
				errorSum += trainingData[1][j] * trainingData[1][j] / 2;
			}

			errorData << i;
			errorData << ", " << errorSum;
			errorData << endl;
			timingData << i << ", " << (endTime - startTime) << endl;
			cout << "Iteration " << i << " " << (endTime - startTime) << "msecs, ETA " << (((double)(endTime - startTime)) * (numberTrainIterations - (double)i) / 1000.0 / 60.0) << "min" << endl;
			if (minTime > (endTime - startTime)) minTime = (endTime - startTime);
			learningRate *= decay;
			accuracyData << i << ", " << (100 * c / (i + 1)) << endl;

			network.toFile(i, numberTrainIterations, decay);
		}
	}

	// load test images
	images = ImageLoader::readMnistImages("/u/trabucco/Desktop/MNIST_Bytes/t10k-images.idx3-ubyte", numberImages, imageSize);
	labels = ImageLoader::readMnistLabels("/u/trabucco/Desktop/MNIST_Bytes/t10k-labels.idx1-ubyte", numberLabels);

	// iterate through each test image
	c = 0;
	for (int i = 0; i < numberTestIterations; i++) {
		startTime = getMSec();
		vector<double> temp = network.classify(images[i]);
		endTime = getMSec();
		if (OutputTarget::getTargetFromOutput(temp) == labels[i]) {
			c += 1;
		}
	}

	accuracyData << numberTrainIterations << ", " << (100 * c / numberTestIterations) << endl;
	cout << endl << "Percentage correct " << (100 * c / numberTestIterations) << "%" << endl;
	cout << "Quickest execution " << minTime << "msecs" << endl;

	cout << "Program finished" << endl;

	errorData.close();
	timingData.close();
	accuracyData.close();

	return 0;
}
