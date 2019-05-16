#pragma warning(disable : 4996)

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <ctime>
#include "neuron.h"
#include "net.h"
#include "function.h"

// default eta , beta
double Neuron::eta = 0.01;     
double Neuron::beta = 0.3;   

// 1000 run average
double NeuralNetwork::m_averageSmoothFactor = 0.999;   


using namespace std;


ostream& operator<<(ostream& os, const Function& func)
{
	for (int i = 0 ; i < func.multipleer.size(); ++i)
	{
		if (i != 0)
			os << "+";
		else
			os << " ";

		if(i == func.multipleer.size()-1)
			os << "(" << func.multipleer[i] <<  ")";
		else
		os  << "("<< func.multipleer[i] << "x^" << func.multipleer.size()-1 - i << ")";
	}
	return os;
}

int maxProb(vector<double> vec)
{
	assert(vec.size() > 0);

	double maxVal = vec[0];
	int maxIndex = 0;
	for (size_t i = 1; i < vec.size(); i++)
	{
		if (maxVal < vec[i]) {
			maxVal = vec[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

struct PredicateResult
{
	double diff;
	int result;
};

PredicateResult predict(vector<double> vec )
{
	assert(vec.size() > 0);

	double maxVal = vec[0];
	double maxVal2 = -1;
	double diff = 0;
	int maxIndex = 0;
	for (size_t i = 1; i < vec.size(); i++)
	{
		if (maxVal < vec[i]) {
			
			maxIndex = i;
			maxVal2 = maxVal;
			maxVal = vec[i];
		}else
		if (maxVal2 < vec[i])
		{
			maxVal2 = vec[i];

		}
		
	}
	PredicateResult res;
	diff = std::min((maxVal - maxVal2) , 1.0);
	res.result = maxIndex;
	res.diff = diff;

	return res;
}
 
void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}
 

void generateRandomFuncs(vector<Function>& funcs , int maxDegree =3, int funcNum = 500, bool verbose = false, bool integerMultipleer = true)
{

	std::default_random_engine gen(time(NULL));
	 

	std::uniform_real_distribution<double> xrand(-100, 100);
	std::uniform_int_distribution<int> arand(0, maxDegree);

	for (size_t i = 0; i < funcNum; i++)
	{
		vector<double> vc;

		for (size_t i = 0; i <= arand(gen); i++)
		{
			auto a = integerMultipleer ? trunc(xrand(gen)) : xrand(gen);
			vc.push_back(a);
		}
		funcs.push_back(Function(vc));
		if(verbose)
		cout << i << ":" << funcs.back() << endl;
	}


}
void generateRandomFuncsStaticDegree(vector<Function>& funcs, int Degree = 3 , int funcNum = 500)
{

	std::default_random_engine gen;
	std::uniform_int_distribution<int> xrand(-10, 10);

	for (size_t i = 0; i < funcNum; i++)
	{
		vector<double> vc;

		for (size_t i = 0; i < Degree; i++)
		{
			auto a = (xrand(gen));
			vc.push_back(a);
		}
		funcs.push_back(Function(vc));

		//cout << i << ":" << funcs.back() << endl;
	}

	cout <<  ":" << " funcs generated "  << endl;
}


void train(NeuralNetwork& net ,int funcNum = 500, int maxDegree = 3, unsigned long long maxIter = 10000 ,double logPercent = 0.01 , bool verbose = false)
{
	vector<Function> funcs;
	double alphaCon = 1.0 - Neuron::beta;
	double etaCon = 0.9;


	

	cout << "generating functions" << endl;
	generateRandomFuncs(funcs, maxDegree, funcNum , verbose);
	cout << "func gen finished" << endl;
	std::default_random_engine gx(time(NULL));
	std::uniform_int_distribution<int> arand(0, funcs.size() - 1);

	int goalPercent = maxIter * logPercent;
	NeuralNetwork::m_averageSmoothFactor = 1.0 - logPercent;

	unsigned long long iter = 0;

	vector<double> inputVals, targetVals;
	int fnc = 0;

	while (iter < maxIter) {
 
		const Function& fn = funcs[arand(gx)];

		fn.getInput(inputVals);
		net.forward(inputVals);
	
		fn.getDegree(targetVals , maxDegree);
		net.backward(targetVals);

		iter++;
		if (iter % goalPercent == 0)
		{
			cout << "iter " << trunc((iter*100.0) / (double)maxIter) << "% avg " << net.getRecentAverageError() << " eta:" <<Neuron::eta << " alpha:" << Neuron::beta << endl;

		}
		if (iter % (int)(maxIter*0.01) == 0)
		{
			Neuron::eta *= etaCon;
		}
	}

	cout << "finished " << endl;

}

void predictFunctionDegree(NeuralNetwork& net,vector<double> vec , int degreeMax )
{
	vector<double> inputVals, targetVals, resultVals;
	Function fx(vec);
	fx.getInput(inputVals);
	net.forward(inputVals);
	fx.getDegree(targetVals , degreeMax);
	net.getResults(resultVals);

	auto res = predict(resultVals);

	cout << "func" << fx << "\t -> Derece:" << maxProb(targetVals) << " Tahmin:" << res.result << (maxProb(targetVals) == res.result ? " +" : " -")  << endl ;


}


void mainDegrees()
{
	Neuron::eta = 0.0000001;
	Neuron::eta = 0.0000171;
	Neuron::beta = 0.997;
	NeuralNetwork net({ 20 ,30, 4 });


	


	auto fileName = "20-30-4_d_i8m_f100k-eta(1.7e-5)-alpha(0.997)-x4.model";
	
	
	bool trainOpt = true;


	if (trainOpt)
	{
		std::chrono::time_point<std::chrono::system_clock> start, end;
		cout << "train start " << endl;
		start = std::chrono::system_clock::now();

		train(net, 100000, 3, 8000000ull);

		end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_seconds = end - start;
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);

		std::cout << "train finished " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << "s\n";

		cout << " avg " << net.getRecentAverageError() << " eta:" << Neuron::eta << " alpha:" << Neuron::beta << endl;
		net.printTopology();
		net.save(fileName);
	}
	else
	{
		net.load(fileName);
		std::cout << "model loaded " << endl;
		cout << " avg " << net.getRecentAverageError() << " eta:" << Neuron::eta << " alpha:" << Neuron::beta << endl;
		net.printTopology();
	}

	cout << "3. derece" << endl;

	predictFunctionDegree(net, {-20, 0,0,15 } ,14);
	predictFunctionDegree(net, { 2,1,6,4 },21);
	predictFunctionDegree(net, { -4,2,-2,10 },3);
	predictFunctionDegree(net, { 1,0,9, -13 },3);

	cout << "2. derece" << endl;

	predictFunctionDegree(net, { 4,3,4 },3);
	predictFunctionDegree(net, { 1,6,4 },3);
	predictFunctionDegree(net, { -4,2,4 },3);
	predictFunctionDegree(net, { -7,-3,9 },3);
	cout << "1. derece" << endl;
	predictFunctionDegree(net, { 20,1 },3);
	predictFunctionDegree(net, { -7,4 },3);
	predictFunctionDegree(net, { 9,36},3);
	predictFunctionDegree(net, { -7,-3 },3);
	cout << "0. derece" << endl;

	predictFunctionDegree(net, { 30 },3);
	predictFunctionDegree(net, { -43 },3);
	predictFunctionDegree(net, { 3 },3);

}

int main()
{
	system("color 0b");

	mainDegrees();


	system("pause");

	return 0;
}
