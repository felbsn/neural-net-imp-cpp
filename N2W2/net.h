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

class NeuralNetwork
{
public:
	NeuralNetwork(const vector<unsigned> &topology);
	NeuralNetwork();
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

	void comp(NeuralNetwork & other);

	void save(const char * file) const;
	void load(const char * file);
	friend Neuron;

	static double m_averageSmoothFactor;
private:
	vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	
};
