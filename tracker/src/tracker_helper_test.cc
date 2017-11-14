#include <cmath>
#include "test_helper.h"
#include "tracker_helper.h"

void test_of_test(void) {
	assert_stoc_eq(1.0, 1.0);
	assert_det_eq(1.0, 1.0);
}

void test_kalman_filter_impl(void) {
	int M = 3;
	
	std::vector<double> o_delta_t(M, 0.1);
	
	std::vector<double> sigma_delta_t(M, 1.0);
	
	std::vector<double> o_delta_beta(M-1, 0.1);
	
	std::vector<double> sigma_delta_beta(M-1, 1.0);
	
	std::vector<double> mu_in(M+1, 0.1);
	
	std::vector<double> p_in((M+1)*(M+1), 0.0);
	for (int i=0; i<M+1; i++) p_in[i*(M+1)+i] = 1.0;
	
	double sigma_z = 1e60;
	double sigma_b = 1e-3;
	
	std::vector<double> mu_out, p_out;
	kalman_filter_impl(sigma_z, sigma_b,
										 &o_delta_t, &sigma_delta_t,
										 &o_delta_beta, &sigma_delta_beta,
										 &mu_in, &p_in, &mu_out, &p_out);
	
	std::vector<double> true_mu = {0.1, 0.2, 0.2, 0.1};
	std::vector<double> true_p = {0.0, 0.0, 0.0, -1.000001};
	for (int i=0; i<M+1; i++)
		assert_det_eq(mu_out[i], true_mu[i]);
	for (int i=0; i<M+1; i++)
		assert_det_eq(p_out[i], true_p[i]);
}

void test_lpf_det(void) {
	int n_vis = 4, n_hid = 4;
	int min_dim = std::fmin(n_vis, n_hid);
	
	std::vector<double> b(n_vis, 0.0);
	std::vector<double> c(n_hid, 0.0);
	std::vector<double> w(n_vis*n_hid, 0.0);
	for (int n=0; n<min_dim; n++)
		w[n*n_hid+n] = 1.0;
	
	double lpf_pred, std_lpf_pred;
	lpf_det(b, c, w, &lpf_pred, &std_lpf_pred);
	double ans = min_dim * log(3.0+exp(1.0));
	assert_det_eq(lpf_pred, ans);
}

void test_lpf_ais(void) {
	int n_vis = 30, n_hid = 20;
	
	std::vector<double> b(n_vis, 0.1);
	std::vector<double> c(n_hid, -0.1);
	std::vector<double> w(n_vis*n_hid, 0.1);
	
	double val_det, std_det;
	lpf_det(b, c, w, &val_det, &std_det);
	double val_ais, std_ais;
	lpf_ais(10, 10000, b, c, w, &val_ais, &std_ais);
	assert_bool(val_ais - 2*std_ais <= val_det);
	assert_bool(val_det <= val_ais + 2*std_ais);
	/*
	std::cout << val_ais << " " << std_ais << std::endl;
	std::cout << val_det << " " << std_det << std::endl;
	*/
}

int main() {
	test_of_test();
	test_kalman_filter_impl();
	test_lpf_det();
	test_lpf_ais();
}