/*
 * Image.h
 *
 *  Created on: Jun 23, 2016
 *      Author: trabucco
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <vector>

using namespace std;

class Image {
private:
	vector<vector<vector<int> > > pixels;
public:
	Image(int x, int y);
	virtual ~Image();
	vector<int> get(int x, int y);
	void set(int x, int y, vector<int> v);
	int getX();
	int getY();
};

#endif /* IMAGE_H_ */
