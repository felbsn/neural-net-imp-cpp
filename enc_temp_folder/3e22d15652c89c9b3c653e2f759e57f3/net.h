#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <fstream>



#include "neuron.h"

using namespace std;

class Net
{
public:
	Net(const vector<unsigned> &topology);
	Net();
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

	void comp(Net & other);

	void save(const char * file) const;
	void load(const char * file);
	friend Neuron;
private:
	vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	static double m_averageSmoothFactor;
};
