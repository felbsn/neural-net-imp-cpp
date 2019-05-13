#pragma once


#include <vector>;
#include <random>

using namespace std;




struct Connection
{
	double weight;
	double deltaWeight;
};


class Neuron;

typedef vector<Neuron> Layer;

 
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	Neuron();
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);


	static double eta;
	static double alpha;

	vector<Connection> m_outputWeights;
	unsigned m_index;

	double m_outputVal;

	double m_gradient;

private:
 
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { 

		static std::default_random_engine gen;
		static std::uniform_real_distribution<double> xrand(-1, 1);

		return xrand(gen);
		//return rand() / double(RAND_MAX); 
	}
	double sumDOW(const Layer &nextLayer) const;

};