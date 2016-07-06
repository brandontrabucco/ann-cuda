/*
 * Image.cpp
 *
 *  Created on: Jun 23, 2016
 *      Author: trabucco
 */

#include "Image.h"

Image::Image(int x, int y) {
	// TODO Auto-generated constructor stub
	for (int i = 0; i < x; i++) {
		vector<vector<int> > temp1;
		for (int j = 0; j < y; j++) {
			vector<int> temp2;
			temp2.push_back(0);
			temp2.push_back(0);
			temp2.push_back(0);
			temp1.push_back(temp2);
		}
		pixels.push_back(temp1);
	}
}

Image::~Image() {
	// TODO Auto-generated destructor stub
}

void Image::set(int x, int y, vector<int> v) {
	pixels[x][y] = v;
}

vector<int> Image::get(int x, int y) {
	return pixels[x][y];
}

int Image::getX() {
	return pixels.size();
}

int Image::getY() {
	return pixels[0].size();
}

