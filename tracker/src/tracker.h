#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <vector>

class Tracker {
	public:
		Tracker(int n_beta, int n_batch, int n_vis, int n_hid);
		Tracker(int n_beta, int n_batch,
						const std::vector<double> &b,
						const std::vector<double> &c,
						const std::vector<double> &w);
		
		void set_params(const std::vector<double> &b,
										const std::vector<double> &c,
										const std::vector<double> &w);
		void get_params(std::vector<double> *b,
										std::vector<double> *c,
										std::vector<double> *w);
		void set_beta_vec(const std::vector<double> &beta_vec);
		void get_beta_vec(std::vector<double> *beta_vec);
		void set_mu(const std::vector<double> &mu);
		void get_mu(std::vector<double> *mu);
		void set_p(const std::vector<double> &p);
		void get_p(std::vector<double> *p);
		void get_estimates(double *lpf, double *std_lpf);
		
		// k: batch index, i: beta index
		void set_particle(int k, int i,
											const std::vector<double> &particle);
		void get_particle(int k, int i, std::vector<double> *particle);
		void run_particle(int k, int i, int n_samp);
		void run_all_particles(int n_samp);
		
		double energy(int i_param, int k, int i_particle);
		double energy(int i_param, int k, int i_particle,
									const std::vector<double> &b,
									const std::vector<double> &c,
									const std::vector<double> &w);
		void prob_swap(int k, int i1, int i2);
		void prob_swap_all(bool odd);
		double log_part_func_det(int i);
		void bridge_sample(const std::vector<double> &o_delta_beta_in,
				               std::vector<double> *o_delta_beta_out,
				               std::vector<double> *sigma_delta_beta);
		void bridge_sample_mean_log(
			  const std::vector<double> &o_delta_beta_in,
		    std::vector<double> *o_delta_beta_out,
		    std::vector<double> *sigma_delta_beta);
		// internal parameters: time step t-1
		// argument parameters: time step t
		// current particles: time step t-1
		void import_sample(const std::vector<double> &b,
											 const std::vector<double> &c,
											 const std::vector<double> &w,
											 std::vector<double> *o_delta_t,
											 std::vector<double> *sigma_delta_t);
		void import_sample_mean_log(
				const std::vector<double> &b,
				const std::vector<double> &c,
				const std::vector<double> &w,
				std::vector<double> *o_delta_t,
				std::vector<double> *sigma_delta_t);
		void init_mu_p(int n_samp, int n_bridge,
									 int n_swap, bool mean_log);
		void init_ais(int n_samp, int n_step);
		void update_mu_p(double sigma_z, double sigma_b,
										 std::vector<double> *o_delta_t,
										 std::vector<double> *sigma_delta_t,
										 std::vector<double> *o_delta_beta,
										 std::vector<double> *sigma_delta_beta);
		
		// calculate part func and its standard deviation of time step t
		// current parameter: time step t-1
		// argument parameters: time step t
		// params after the function execution will be the argument params
		void track(const std::vector<double> &b,
							 const std::vector<double> &c,
							 const std::vector<double> &w,
							 int n_samp_t, int n_samp_beta, int n_bridge,
							 double sigma_z, double sigma_b, bool mean_log);
		
	private:
		int _n_beta;
		int _n_batch;
		int _n_vis, _n_hid;
		
		void _init(int n_beta, int n_batch,
							 const std::vector<double> &b,
							 const std::vector<double> &c,
							 const std::vector<double> &w);
		
		// first element must be 1.0, last element must be 0.0, decreasing
		std::vector<double> _beta_vec;
		// meaning of dimension 0: batch, 1: beta, 2: activations
		// store _n_vis + _n_hid activations in this order
		std::vector<std::vector<std::vector<double> > > _particles;
		// parameters,
		// element at visible index n hidden index m is _w[n*_n_hid+m]
		std::vector<double> _b, _c, _w;
		// _mu is a vector of size _n_beta + 1, last elemet is bias
		// element (i, j) is stored in _p[i*(_n_beta+1)+j]
		std::vector<double> _mu, _p;
};

#endif // _TRACKER_H_
