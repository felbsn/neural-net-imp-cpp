#pragma once

#include <vector>


using namespace std;

class Function
{
public:
	Function();
	Function(vector<double> values);
	double solve(double x) const;

	void add(double multipleer);

	void getDegree(vector<double>& degrees , int maxDeg = 3) const;

	void getInput(vector<double>& inputs, int num = 20) const;

	void getInputJ(vector<double>& inputs, int num = 20) const;
	void getTargetValues(vector<double>& inputs, double normMax = 0) const;


	int getDegree() const { return multipleer.size() - 1; }
	double getMultipleer(int at) const { return multipleer[at]; }
	double getMaxMultipleer() const;

	friend ostream& operator<<(ostream& os, const Function& dt);


	vector< double >  multipleer;
private:

	
	
};



