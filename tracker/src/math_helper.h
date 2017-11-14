#ifndef _MATH_HELPER_H_
#define _MATH_HELPER_H_

#include <random>

extern std::mt19937 gen;
extern std::uniform_real_distribution<double> uni_dist;

double bin_dist(double p);
double sigmoid(double x);
double clip(double x);

#endif // _MATH_HELPER_H_