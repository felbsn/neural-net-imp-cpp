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

double Neuron::eta = 0.01;     
double Neuron::alpha = 0.3;   

double Net::m_averageSmoothFactor = 0.999;   


using namespace std;


#define SCALE 100



ostream& operator<<(ostream& os, const Function& func)
{
	for (int i = 0 ; i < func.multipleer.size(); ++i)
	{
		if (i != 0)
			os << "+";
		else
			os << " ";

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

PredicateResult maxProbD(vector<double> vec )
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


void train(Net& net ,int funcNum = 500, int maxDegree = 3, unsigned long long maxIter = 10000 ,double logPercent = 0.01 , bool verbose = false)
{
	vector<Function> funcs;
	double alphaCon = 1.0 - Neuron::alpha;
	double etaCon = 0.9;

	cout << "generating functions" << endl;
	generateRandomFuncs(funcs, maxDegree, funcNum , verbose);
	cout << "func gen finished" << endl;
	std::default_random_engine gx(time(NULL));
	std::uniform_int_distribution<int> arand(0, funcs.size() - 1);

	int goalPercent = maxIter * logPercent;
	unsigned long long iter = 0;

	vector<double> inputVals, targetVals;
	int fnc = 0;

	while (iter < maxIter) {
 
		const Function& fn = funcs[arand(gx)];

		fn.getInput(inputVals);
		net.feedForward(inputVals);
	
		fn.getDegree(targetVals , maxDegree);
		net.backProp(targetVals);

		iter++;
		if (iter % goalPercent == 0)
		{
			cout << "iter " << trunc((iter*100.0) / (double)maxIter) << "% avg " << net.getRecentAverageError() << " eta:" <<Neuron::eta << " alpha:" << Neuron::alpha << endl;

			//Neuron::alpha += alphaCon * (double)iter / (double)maxIter*0.97;
			//Neuron::alpha = std::min(0.999999, Neuron::alpha);
		
			

		}
		if (iter % (int)(maxIter*0.01) == 0)
		{
			Neuron::eta *= etaCon;
		}
	}

	cout << "finished " << endl;

}
void trainEx(Net& net, int funcNum = 500, int Degree = 3, int maxIter = 10000)
{
	std::default_random_engine gen;
	std::uniform_real_distribution<double> urand(0, 1);

	vector<Function> funcs;
	generateRandomFuncsStaticDegree(funcs, Degree, funcNum);

	vector<double> inputVals, targetVals;
	int max_iter = maxIter;
	int iter = 0;



	double oldError = 0;
	while (iter < max_iter) {
		int fnc = urand(gen) *(funcs.size() - 1);

		funcs[fnc].getInput(inputVals);
		net.feedForward(inputVals);

		funcs[fnc].getTargetValues(targetVals);
		net.backProp(targetVals);

		double deltaErr = net.getRecentAverageError() - oldError;
		/*if (deltaErr < 0)
		{
			Neuron::eta += 0.01;
			Neuron::eta = std::min(0.6, Neuron::eta);

			Neuron::alpha -= 0.01;
			Neuron::alpha = std::max(0.1, Neuron::alpha);

		}
		else
		{
			Neuron::eta -= 0.01;
			Neuron::eta = std::max(0.01, Neuron::eta);

			Neuron::alpha += 0.01;
			Neuron::alpha = std::min(0.9, Neuron::alpha);
		}*/
			
		oldError = net.getRecentAverageError();
		//double Neuron::eta = 0.01;
		//double Neuron::alpha = 0.3;

		if (iter % 1000 == 0)
			cout << "iter " << iter << " avg " << net.getRecentAverageError()  << " eta:" << Neuron::eta <<" alpha:" << Neuron::alpha << endl;
		iter++;
	}
}

void trainExU(Net& net, vector<double> vec, int maxIter = 10000)
{
	vector<double> inputVals, targetVals;
	Function fn(vec);
	int iter = 0;
	while (iter < maxIter) {
		 
		fn.getInputJ(inputVals);
		net.feedForward(inputVals);

		fn.getTargetValues(targetVals);
		net.backProp(targetVals);

		if (iter % 1000 == 0)
		{
			cout << "iter " << iter << " avg " << net.getRecentAverageError() << " eta:" << Neuron::eta << " alpha:" << Neuron::alpha << endl;
		}
		iter++;
	}
}
void trainExU(Net& net, Function& fn, int maxIter = 10000 , double normMax = 0)
{
	vector<double> inputVals, targetVals;
	int iter = 0;
	int twoPercent = maxIter * 0.02;
	while (iter < maxIter) {

		fn.getInputJ(inputVals);
		net.feedForward(inputVals);

		fn.getTargetValues(targetVals , normMax);
		net.backProp(targetVals);

		if (iter % twoPercent == 0)
		{
			cout << "iter " << trunc( (iter*100.0)/(double) maxIter)<< "% avg " << net.getRecentAverageError() << " eta:" << Neuron::eta << " alpha:" << Neuron::alpha << endl;
		}
		iter++;
	}
}

void predictDegree(Net& net,vector<double> vec , int degreeMax )
{
	vector<double> inputVals, targetVals, resultVals;
	Function fx(vec);
	fx.getInput(inputVals);
	net.feedForward(inputVals);
	fx.getDegree(targetVals , degreeMax);
	net.getResults(resultVals);

	cout << "func" << fx << endl;

	auto res = maxProbD(resultVals);
	cout << "target: " << maxProb(targetVals) << endl;
	cout << "Result: " << res.result << " dif:" << res.diff << endl;
}

void predictValues(Net& net, vector<double> vec , double normMax = 0)
{
	vector<double> inputVals, targetVals, resultVals;
	Function fx(vec);
	fx.getInput(inputVals);
	net.feedForward(inputVals);
	net.getResults(resultVals);

	cout << "func" << fx << endl;

	fx.getTargetValues(targetVals , normMax);
	showVectorVals("res:", resultVals);
	showVectorVals("targets:", targetVals);
}


void mainValues()
{
	Neuron::eta = 0.018;
	Neuron::alpha = 0.01;
	Net net({ 20 ,20,16, 2 });


	vector<Function> funcs;
	generateRandomFuncsStaticDegree(funcs, 2, 5000);

	for (size_t i = 0; i < funcs.size(); i++)
	{
		
		trainExU(net, funcs[i], 20 , 60);
	}


	predictValues(net, funcs[4].multipleer, 60);

	//net.save("20-100-2 v 1m.model");

	predictValues(net, { 4,2 } ,60);
	predictValues(net, { -1,2 }, 60);
	predictValues(net, { 2,-2 }, 60);

	predictValues(net, { 16,-23 }, 60);
	predictValues(net, { 40,-3 }, 60);
}

void mainDegrees()
{
	Neuron::eta = 0.0000001;
	Neuron::eta = 0.0000071;
	Neuron::alpha = 0.995;
	Net net({ 20 ,30, 4 });


	std::chrono::time_point<std::chrono::system_clock> start, end;
	cout << "train start " << endl;
	auto fileName = "20-30-4_d_i30m_f100k-eta(6e-7)-alpha(0.995)-x0.model";
	bool trainOpt = true;


	if (trainOpt)
	{
		start = std::chrono::system_clock::now();
		//30000000ull
		train(net, 100000, 3, 30000000ull);
		//train(net, 100000, 3, 1000000ull);
		end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_seconds = end - start;
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);

		std::cout << "train finished " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << "s\n";
		//std::cout << "train finished " << endl;


		net.save(fileName);
	}
	else
	{
		net.load(fileName);
	}

    
	//net.load("20-16-10-3x.model");

	cout << "3. derece" << endl;

	predictDegree(net, {6, 4,3,4 } ,3);
	predictDegree(net, { 2,1,6,4 },3);
	predictDegree(net, { -4,2,-2,10 },3);
	predictDegree(net, { 1,0,9,3 },3);

	cout << "2. derece" << endl;

	predictDegree(net, { 4,3,4 },3);
	predictDegree(net, { 1,6,4 },3);
	predictDegree(net, { -4,2,4 },3);
	predictDegree(net, { -7,-3,9 },3);
	cout << "1. derece" << endl;
	predictDegree(net, { 20,1 },3);
	predictDegree(net, { -7,4 },3);
	predictDegree(net, { 9,36},3);
	predictDegree(net, { -7,-3 },3);
	cout << "0. derece" << endl;

	predictDegree(net, { 30 },3);
	predictDegree(net, { -43 },3);
	predictDegree(net, { 3 },3);

}

int main()
{

	

	//mainValues();
	mainDegrees();



	system("pause");

	return 0;
}

/*
	Net net;
	net.load("20-16-10-3x.model");


	{
		vector<double> inputVals, targetVals, resultVals;
		Function fx({ -3 , -1 ,-4 });
		fx.getInput(inputVals);
		net.feedForward(inputVals);
		fx.getDegree(targetVals);
		net.getResults(resultVals);

		cout << "func" << fx << endl;

		auto res = maxProbD(resultVals);
		cout << "target: " << maxProb(targetVals) << endl;
		cout << "Result: " << res.result << " dif:"  << res.diff<< endl;

	}





	{
		vector<double> inputVals, targetVals, resultVals;
		Function fx({13 });
		fx.getInput(inputVals);
		net.feedForward(inputVals);
		fx.getDegree(targetVals);
		net.getResults(resultVals);

		cout << "func" << fx << endl;

		auto res = maxProbD(resultVals);
		cout << "target: " << maxProb(targetVals) << endl;
		cout << "Result: " << res.result << " dif:" << res.diff << endl;

	}

	system("pause");


*/