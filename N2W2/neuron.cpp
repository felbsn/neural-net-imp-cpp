#include "neuron.h"



void Neuron::updateInputWeights(Layer &prevLayer)
{

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

		double newDeltaWeight =
		// training rate                             momentum 
			eta* neuron.getOutputVal()* m_gradient + alpha* oldDeltaWeight;

		neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_index].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	// toplam hata ->

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);//m_outputVal
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]

	//f = 1/(1+exp(-x)) sigmoid
	//return 1.0 / (1.0 +  exp(-x));
	

	//return x;
	//tanh
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{

	// df = f * (1 - f) sigmoid derivative
	//double f = 1.0 / (1.0 + exp(-x));
	//double df = f * (1.0 - f);
	//return df;


	// tanh derivative
	return 1.0 - tanh(x) * tanh(x);
	//return 1.0 - (x) * (x);
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// ileri iterasyon

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_index].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_index = myIndex;
}

Neuron::Neuron()
{
}
