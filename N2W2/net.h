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
	void forward(const vector<double> &inputVals);
	void backward(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

	void printTopology()
	{
		cout << "Topology ->" << endl;

		for (size_t i = 0; i < m_layers.size(); i++)
		{
			cout << m_layers[i].size() - 1 << endl;
		}

		cout << "____" << endl;
	}

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
