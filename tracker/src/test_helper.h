#ifndef _TEST_HELPER_H_
#define _TEST_HELPER_H_

#include <iostream>
#include <cassert>

#define STOC_DELTA 1e-5
#define DET_DELTA 1e-7

#define assert_stoc_eq(a, b) assert_stoc_eq_impl(a, b, __LINE__)
void assert_stoc_eq_impl(double a, double b, int line) {
  if (!(a - STOC_DELTA < b && b < a + STOC_DELTA)) {
    std::cout << "Error assert_stoc_eq of line " << line << std::endl;
    std::cout <<
      "This might be because of the stochasticity of test itself"
      << std::endl;
  }
}

#define assert_det_eq(a, b) assert_det_eq_impl(a, b, __LINE__)
void assert_det_eq_impl(double a, double b, int line) {
  if (!(a - DET_DELTA < b && b < a + DET_DELTA)) {
    std::cout << "Error assert_det_eq of line " << line << std::endl;
  }
}

#define assert_bool(flag) assert_bool_impl(flag, __LINE__)
void assert_bool_impl(bool flag, int line) {
  if (!flag) {
    std::cout << "Error assert of line " << line << std::endl;
  }
}

#endif // _TEST_HELPER_H_
