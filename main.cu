//============================================================================
// Name        : main.cpp
// Author      : Brandon Trabucco
// Version     : 1.0.5
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

#define CONVERGENCE_TEST false
#define TEST_ACCURACY true

using namespace std;

long long getMSec() {
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
	// start program and validate input
	cout << "Program initializing" << endl;
	if (argc < 6) {
		cout << argv[0] << " <o | b> <training size> <repeat size> <learning rate> <decay rate> <size ...>" << endl;
		return -1;
	}


	/**
	 *
	 * 	Declare the global variables
	 * 	These govern functionality of the program
	 *
	 */
	vector<int> size;
	vector<vector<double> > trainingImages, testImages;
	vector<double> trainingLabels, testLabels;
	int numberImages = 0;
	int imageSize = 0;
	int numberLabels = 0;
	int trainingSize = atoi(argv[2]);
	int repeatImages = atoi(argv[3]);
	int testSize = 1000;
	int updatePoints = (repeatImages < 100) ? repeatImages : 100;
	int savePoints = (repeatImages < 10) ? repeatImages : 10;
	double learningRate = atof(argv[4]), decay = atof(argv[5]);
	long long networkStart, networkEnd, sumTime = 0, iterationStart;
	bool enableBatch = (argv[1][0] == 'b');


	/**
	 *
	 * 	Open file streams to save data samples from Neural Network
	 * 	This data can be plotted via GNUPlot
	 *
	 */
	ostringstream errorDataFileName;
	errorDataFileName << "/u/trabucco/Desktop/MNIST_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << getDate()->tm_mday <<
			"_GPU-ANN-"  << (enableBatch ? "Batch" : "Incremental") << "-Error_" <<
			(trainingSize * repeatImages) <<
			"-iterations_" << repeatImages <<
			"-repeat_" << learningRate <<
			"-learning_" << decay << "-decay.csv";
	ostringstream timingDataFileName;
	timingDataFileName << "/u/trabucco/Desktop/MNIST_Data_Files/GPU-" << (enableBatch ? "Batch" : "Incremental") << "-Timing.csv";
	ostringstream overheadDataFileName;
	overheadDataFileName << "/u/trabucco/Desktop/MNIST_Data_Files/GPU-" << (enableBatch ? "Batch" : "Incremental") << "-Overhead.csv";
	ostringstream accuracyDataFileName;
	accuracyDataFileName << "/u/trabucco/Desktop/MNIST_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << getDate()->tm_mday <<
			"_GPU-ANN-" << (enableBatch ? "Batch" : "Incremental") << "-Accuracy_" <<
			(trainingSize * repeatImages) <<
			"-iterations_" << repeatImages <<
			"-repeat_" << learningRate <<
			"-learning_" << decay << "-decay.csv";

	ofstream errorData(errorDataFileName.str());
	if (!errorData.is_open()) return -1;
	ofstream timingData(timingDataFileName.str(), ios::app);
	if (!timingData.is_open()) return -1;
	ofstream overheadData(overheadDataFileName.str(), ios::app);
	if (!overheadData.is_open()) return -1;
	ofstream accuracyData(accuracyDataFileName.str());
	if (!accuracyData.is_open()) return -1;


	/**
	 *
	 * 	Load the MNIST Dataset
	 * 	To feed into Neural Network
	 *
	 */
	// load training images and labels
	networkStart = getMSec();
	trainingImages = ImageLoader::readMnistImages("/u/trabucco/Desktop/MNIST_Bytes/train-images.idx3-ubyte", numberImages, imageSize);
	trainingLabels = ImageLoader::readMnistLabels("/u/trabucco/Desktop/MNIST_Bytes/train-labels.idx1-ubyte", numberLabels);
	// load test images and labels
	testImages = ImageLoader::readMnistImages("/u/trabucco/Desktop/MNIST_Bytes/t10k-images.idx3-ubyte", numberImages, imageSize);
	testLabels = ImageLoader::readMnistLabels("/u/trabucco/Desktop/MNIST_Bytes/t10k-labels.idx1-ubyte", numberLabels);
	networkEnd = getMSec();
	cout << "Training images and labels loaded in " << (networkEnd - networkStart) << " msecs" << endl;


	/**
	 *
	 * 	Set the size of the Neural Network
	 * 	Each element is one layer
	 *
	 */
	size.push_back(imageSize);
	for (int i = 0; i < (argc - 6); i++) {
		size.push_back(atoi(argv[i + 6]));
	}
	size.push_back(10);


	/**
	 *
	 * 	Initialize the Neural Network
	 * 	Parameters are set
	 *
	 */
	NeuralNetwork network = NeuralNetwork(size, 1.0, learningRate, false);


	/**
	 *
	 * 	Iterate through the training and test datasets
	 * 	Output performance information to data files
	 *
	 */
	if (!enableBatch) {
		/**
		*
		* 	This section is for incremental gradient descent
		*
		*/
		for (int r = 0; r < repeatImages; r++) {
			for (int i = 0; i < trainingSize; i++) {
				int absoluteIteration = (r * trainingSize) + i;
				iterationStart = getMSec();

				networkStart = getMSec();
				vector<vector<double> > trainingData = network.online(trainingImages[CONVERGENCE_TEST ? 0 : i], OutputTarget::getTargetOutput(trainingLabels[CONVERGENCE_TEST ? 0 : i]), learningRate, (!(r % (repeatImages / updatePoints))) && (i == (trainingSize - 1)));
				networkEnd = getMSec();
				sumTime += (networkEnd - networkStart);
				if (!(r % (repeatImages / updatePoints)) && (i == (trainingSize - 1)) && TEST_ACCURACY) {
					errorData << absoluteIteration;
					errorData << ", " << trainingData[1][0];
					errorData << endl;
					learningRate *= decay;

					// iterate through each test image to get current accuracy
					int c = 0;
					for (int j = 0; j < testSize; j++) {
						networkStart = getMSec();
						vector<double> temp = network.classify(testImages[j]);
						networkEnd = getMSec();
						if (OutputTarget::getTargetFromOutput(temp) == testLabels[j]) {
							c += 1;
						}
					} accuracyData << absoluteIteration << ", " << (100 * c / testSize) << endl;
					cout << "Iteration " << absoluteIteration << " " << (((getMSec() - iterationStart) / updatePoints) + (networkEnd - networkStart)) << "msecs, ETA " << (((double)(((getMSec() - iterationStart) / updatePoints) + (networkEnd - networkStart))) * ((trainingSize * repeatImages) - (double)absoluteIteration) / 1000.0 / 60.0) << "min" << endl;
				} if (!(r % (repeatImages / savePoints)) && (i == (trainingSize - 1)) && TEST_ACCURACY) {
					network.toFile(absoluteIteration, trainingSize, repeatImages, decay);
				}
			}
		}
	} else {
		/**
		 *
		 * 	This section is for batch gradient descent
		 *
		 */
		for (int r = 0; r < repeatImages; r++) {
			for (int i = 0; i < trainingSize; i++) {
				int absoluteIteration = (r * trainingSize) + i;
				iterationStart = getMSec();

				networkStart = getMSec();
				vector<vector<double> > trainingData = network.batch(trainingImages[CONVERGENCE_TEST ? 0 : i], OutputTarget::getTargetOutput(trainingLabels[CONVERGENCE_TEST ? 0 : i]), learningRate, (!(r % (repeatImages / updatePoints)) && (i == (trainingSize - 1))), (i == (trainingSize - 1)));
				networkEnd = getMSec();
				sumTime += (networkEnd - networkStart);
				if (!(r % ((repeatImages) / updatePoints)) && (i == (trainingSize - 1)) && TEST_ACCURACY) {
					errorData << absoluteIteration;
					errorData << ", " << trainingData[1][0];
					errorData << endl;
					learningRate *= decay;

					// iterate through each test image to get current accuracy
					int c = 0;
					for (int j = 0; j < testSize; j++) {
						networkStart = getMSec();
						vector<double> temp = network.classify(testImages[j]);
						networkEnd = getMSec();
						if (OutputTarget::getTargetFromOutput(temp) == testLabels[j]) {
							c += 1;
						}
					} accuracyData << absoluteIteration << ", " << (100 * c / testSize) << endl;
					cout << "Iteration " << absoluteIteration << " " << (((getMSec() - iterationStart) / updatePoints) + (networkEnd - networkStart)) << "msecs, ETA " << (((double)(((getMSec() - iterationStart) / updatePoints) + (networkEnd - networkStart))) * ((trainingSize * repeatImages) - (double)absoluteIteration) / 1000.0 / 60.0) << "min" << endl;
				} if (!(r % (repeatImages / savePoints)) && (i == (trainingSize - 1)) && TEST_ACCURACY) {
					network.toFile(absoluteIteration, trainingSize, repeatImages, decay);
				}
			}
		}
	}


	// end program and close file streams
	timingData << (accumulate(size.begin(), size.end(), 0)) << ", " << sumTime << endl;
	overheadData << (accumulate(size.begin(), size.end(), 0)) << ", " << (((double)NeuralNetwork::overhead)/((double)NeuralNetwork::computation)) << endl;
	cout << "Total computation time " << sumTime << "msecs" << endl;
	cout << "Program finished" << endl;

	errorData.close();
	timingData.close();
	overheadData.close();
	accuracyData.close();

	return 0;
}
