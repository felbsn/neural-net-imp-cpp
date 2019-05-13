
#include "function.h"
#include <cassert>

Function::Function()
{
}

Function::Function(vector<double> values)
{
	//multipleer.resize(values.size());
	multipleer.clear();
	copy(values.begin(), values.end(),back_inserter(multipleer) );
	while (multipleer.size() > 0 &&  multipleer[0] == 0)
	{
		multipleer.erase(multipleer.begin());
	}
	if (multipleer.size() == 0)
	{
		multipleer.push_back(1);
	}
}

double Function::solve(double x) const
{
	double sum = 0;
	for (int i = 0; i < multipleer.size(); i++)
	{
		sum += multipleer[i] * pow(x, i);
	}

	return sum;
}

void Function::add(double multipleer)
{
	this->multipleer.push_back(multipleer);
}


void Function::getDegree(vector<double> &degrees , int maxDegree) const
{
	degrees.clear();
	for (int i = 0; i <= maxDegree; i++)
	{
		degrees.push_back(-1.0);
		 
	}
	//assert(multipleer.size()-1 <= maxDegree);
	degrees[multipleer.size() -1 ] = 1.0;
}

void Function::getInput(vector<double> &inputs, int num ) const
{
	inputs.clear();

	double start = num * -5 * num;

	double maxv;
	double maxx;

	maxv = abs(solve(start));


	inputs.push_back(solve(start));

	start += 10;

	for (int i = 1; i < num; i++)
	{
		maxv = std::fmax(maxv, abs(solve(start)));

		inputs.push_back(solve(start));

		start += 10;
	}

	if (maxv == 0)
	{

		return;
	}
	for (int i = 0; i < num; i++)
	{

		inputs[i] /= maxv;


	}
}

void Function::getInputJ(vector<double> &inputs, int num ) const
{
	inputs.clear();

	double adv = (rand() / RAND_MAX)*30;

	double start = num * -adv *0.5 * num;

	double maxv;
	double maxx;

	maxv = abs(solve(start));


	inputs.push_back(solve(start));

	start += adv;

	for (int i = 1; i < num; i++)
	{
		maxv = std::fmax(maxv, abs(solve(start)));

		inputs.push_back(solve(start));

		start += adv;
	}

	if (maxv == 0)
	{

		return;
	}
	for (int i = 0; i < num; i++)
	{

		inputs[i] /= maxv;


	}
}

void Function::getTargetValues(vector<double> &inputs , double normMax ) const
{
	inputs.clear();



	std::copy(multipleer.begin(), multipleer.end(), back_inserter(inputs));
	double maxX;
	if (normMax == 0)
		maxX = getMaxMultipleer();
	else
		maxX = normMax;

	for (size_t i = 0; i < inputs.size(); i++)
	{
		inputs[i] /= maxX;
	}

}

double Function::getMaxMultipleer() const
{
	double maxv = std::abs(multipleer[0]);
	for (size_t i = 1; i < multipleer.size(); i++)
	{
		if ( std::abs(multipleer[i]) > maxv) maxv = std::abs(multipleer[i]);
	}

	// cant let 0 ...
	//if (maxv == 0) return 0.000001;
	return maxv;
}

 

