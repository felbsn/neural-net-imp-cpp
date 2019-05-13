
#include "net.h"



void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}


#define log(x) 
// cout << x << endl;
void Net::comp(Net& other)
{
	

	for (size_t i = 0; i < m_layers.size(); i++)
	{
		for (size_t j = 0; j < m_layers[i].size(); j++)
		{
			for (size_t k = 0; k < m_layers[i][j].m_outputWeights.size(); k++)
			{
				if (m_layers[i][j].m_outputWeights[k].weight != other.m_layers[i][j].m_outputWeights[k].weight)
				{
					cout << "not equal in i:" << i << " j:" << j << " k:" << k << " >" << m_layers[i][j].m_outputWeights[k].weight << "/" <<
						other.m_layers[i][j].m_outputWeights[k].weight << endl;
				}
			}
		}

	}
	

}

void Net::save(const char * file) const
{
	log("writing " << file);
	auto myfile = std::fstream(file, std::ios::out | std::ios::binary);
	int topologySize = m_layers.size();

	myfile.write( (char*)&topologySize ,  sizeof(int));
	log("top size " << topologySize);
	for (size_t i = 0; i < m_layers.size(); i++)
	{
		int s = m_layers[i].size();
		myfile.write((char*)&s, sizeof(int));

		log("t " << i <<" " << s);
	}
	log(" writing layers " );
	for (size_t i = 0; i < m_layers.size(); i++)
	{
		int s = m_layers[i].size();
		const Layer &layer = m_layers[i];
		for (size_t j = 0; j < layer.size(); j++)
		{
		
			const auto& n = layer[j];

			int outputCount = n.m_outputWeights.size();
			myfile.write((char*)&outputCount, sizeof(int));

			log(" writing n outputs:" << outputCount);


			int nindex = n.m_index;
			myfile.write((char*)&nindex, sizeof(int));

			log(" writing n index:" << nindex);

			myfile.write((char*)&n.m_outputVal, sizeof(double));

			myfile.write((char*)&n.m_gradient, sizeof(double));

			for (size_t k = 0; k < n.m_outputWeights.size(); k++)
			{
				myfile.write((char*)&n.m_outputWeights[k], sizeof(Connection));
			}

			for (size_t k = 0; k < n.m_outputWeights.size(); k++)
			{
				//cout << " w:" << n.m_outputWeights[k].weight;
			}

			log(" neuron writed. ");
		}
	}



	myfile.close();


}

void Net::load(const char * file)
{
	log("reading " << file);
	auto myfile = std::fstream(file, std::ios::in | std::ios::binary);
	int topologySize;

	myfile.read((char*)&topologySize, sizeof(int));

	m_layers.resize(topologySize);
	log("readed top size " << topologySize);

	for (size_t i = 0; i < m_layers.size(); i++)
	{
		int s;
		myfile.read((char*)&s, sizeof(int));
		m_layers[i].resize(s);

		log("t " << i   << " --- "<< s);
	}

	log(" reading layers ");

	for (size_t i = 0; i < m_layers.size(); i++)
	{
		Layer &layer = m_layers[i];


		for (size_t j = 0; j < layer.size(); j++)
		{
			auto& n = layer[j];

			int outputCount;
			myfile.read((char*)&outputCount, sizeof(int));
			n.m_outputWeights.resize(outputCount);

			log(" neuron weighs "  << outputCount);

			int nindex;
			myfile.read((char*)&nindex, sizeof(int));
			n.m_index = nindex;

			log(" neuron index " << nindex);


			myfile.read((char*)&n.m_outputVal, sizeof(double));

			myfile.read((char*)&n.m_gradient, sizeof(double));

			for (size_t k = 0; k < n.m_outputWeights.size(); k++)
			{
				myfile.read((char*)&n.m_outputWeights[k], sizeof(Connection));
			}

			for (size_t k = 0; k < n.m_outputWeights.size(); k++)
			{
				//cout << " w:" << n.m_outputWeights[k].weight;
			}

			log(" neuron read. ");
		}
	}

	myfile.close();
}

void Net::backProp(const vector<double> &targetVals)
{
	// Calculate overall net error (RMS of output neuron errors)


	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	if (!(targetVals.size() == outputLayer.size() - 1))
	{
		cout << "target" << targetVals.size() << endl;
		cout << "output" << outputLayer.size() << endl;

	}


	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement

	m_recentAverageError = m_recentAverageError * m_averageSmoothFactor + (1.0 - m_averageSmoothFactor)*m_error;
 

	// Calculate output layer gradients

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate hidden layer gradients

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	if (inputVals.size() != m_layers[0].size() - 1)
	{
		auto msg = "Input degerleri ve topoloji uyumsuz! " + to_string(inputVals.size()) + " / " + to_string(m_layers[0].size() - 1);
		throw exception(msg.c_str());
	}


	// inputları gir
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// ileri iterasyon
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];

		// bias noronu dahil degil -1
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have a new layer, now fill it with neurons, and
		// add a bias neuron in each layer.
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			 
		}

		// Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
		m_layers.back().back().setOutputVal(1.0);
	}
}

Net::Net()
{
}
