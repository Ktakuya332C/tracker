#ifndef _TRACKER_HELPER_H_
#define _TRACKER_HELPER_H_

#include <vector>

void kalman_filter_impl(double sigma_z, double sigma_b,
												std::vector<double> *o_delta_t,
												std::vector<double> *sigma_delta_t,
												std::vector<double> *o_delta_beta,
												std::vector<double> *sigma_delta_beta,
												std::vector<double> *mu_in,
												std::vector<double> *p_in,
												std::vector<double> *mu_out,
												std::vector<double> *p_out);

void lpf_ais(int n_samp, int n_step,
						 const std::vector<double> &b,
						 const std::vector<double> &c,
						 const std::vector<double> &w,
						 double *lpf, double *std_lpf);

void lpf_det(const std::vector<double> &b,
						 const std::vector<double> &c,
						 const std::vector<double> &w,
						 double *lpf, double *std_lpf);

#endif // _TRACKER_HELPER_H_