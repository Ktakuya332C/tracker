#include "math_helper.h"
#include <cmath>

const double DOUBLE_LOWER_BOUND = -1e20;
const double DOUBLE_UPPER_BOUND = 1e20;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

double bin_dist(double p) {
  if (p > uni_dist(gen)) {
    return 1.0;
  } else {
    return 0.0;
  }
}

double sigmoid(double x) {
  if (x > 0) {
    return 1 / (1 + exp(-x));
  } else {
    return exp(x) / (1 + exp(x));
  }
}

double clip(double x) {
  return std::max(DOUBLE_LOWER_BOUND, std::min(x, DOUBLE_UPPER_BOUND));
}