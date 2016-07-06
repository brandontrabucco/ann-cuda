/*
 * ImageLoader.h
 *
 *  Created on: Jun 23, 2016
 *      Author: trabucco
 */

#ifndef IMAGELOADER_H_
#define IMAGELOADER_H_

#include "Image.h"
#include <string.h>
#include <vector>
#include <uchar.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace std;

class ImageLoader {
private:
	static int reverseInt(int i);
public:
	static vector<vector<double> > readMnistImages(const char full_path[], int& number_of_images, int& image_size);
	static vector<double> readMnistLabels(const char full_path[], int& number_of_labels);
};

#endif /* IMAGELOADER_H_ */
