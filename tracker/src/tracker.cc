#include "tracker.h"
#include <iostream>
#include <cassert>
#include "math_helper.h"
#include "tracker_helper.h"

Tracker::Tracker(int n_beta, int n_batch, int n_vis, int n_hid) {
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, 0.0);
  std::vector<double> w(n_vis*n_hid, 0.0);
  _init(n_beta, n_batch, b, c, w);
}

Tracker::Tracker(int n_beta, int n_batch,
                 const std::vector<double> &b,
                 const std::vector<double> &c,
                 const std::vector<double> &w) {
  _init(n_beta, n_batch, b, c, w);
}

void Tracker::_init(int n_beta, int n_batch,
                    const std::vector<double> &b,
                    const std::vector<double> &c,
                    const std::vector<double> &w) {
  _n_beta = n_beta;
  assert(_n_beta >= 2);
  _n_batch = n_batch;
  assert(_n_batch > 0);
  _n_vis = b.size();
  _n_hid = c.size();
  assert(w.size() == _n_vis * _n_hid);
  
  _b = b;
  _c = c;
  _w = w;
  
  // initialize _particles
  for (int k=0; k<_n_batch; k++) {
    _particles.push_back(std::vector<std::vector<double> >());
    for (int i=0; i<_n_beta; i++) {
      _particles[k].push_back(std::vector<double>());
      _particles[k][i].resize(_n_vis + _n_hid);
      for (int n=0; n<_n_vis+_n_hid; n++)
        _particles[k][i][n] = bin_dist(0.5);
    }
  }
  
  // initialize _beta_vec
  _beta_vec.resize(_n_beta);
  for (int i=0; i<_n_beta; i++) {
    _beta_vec[i] = 1.0 - (double)i / (double)(_n_beta-1);
  }
}

void Tracker::set_params(const std::vector<double> &b,
                         const std::vector<double> &c,
                         const std::vector<double> &w) {
  assert(b.size() == _n_vis);
  assert(c.size() == _n_hid);
  assert(w.size() == _n_vis * _n_hid);
  
  _b = b;
  _c = c;
  _w = w;
}

void Tracker::get_params(std::vector<double> *b,
                         std::vector<double> *c,
                         std::vector<double> *w) {
  *b = _b;
  *c = _c;
  *w = _w;
}

void Tracker::set_beta_vec(const std::vector<double> &beta_vec) {
  assert(beta_vec.size() == _n_beta);
  _beta_vec = beta_vec;
}

void Tracker::get_beta_vec(std::vector<double> *beta_vec) {
  *beta_vec = _beta_vec;
}

void Tracker::set_mu(const std::vector<double> &mu) {
  assert(mu.size() == _n_beta+1);
  _mu = mu;
}

void Tracker::get_mu(std::vector<double> *mu) {
  *mu = _mu;
}

void Tracker::set_p(const std::vector<double> &p) {
  assert(p.size() == (_n_beta+1)*(_n_beta+1));
  _p = p;
}

void Tracker::get_p(std::vector<double> *p) {
  *p = _p;
}

void Tracker::get_estimates(double *lpf, double *std_lpf) {
  // suppose already initialized
  assert(_mu.size() == _n_beta+1);
  assert(_p.size() == (_n_beta+1)*(_n_beta+1));
  
  *lpf = _mu[0];
  *std_lpf = sqrt(_p[0]);
}

void Tracker::set_particle(int k, int i,
                           const std::vector<double> &particle) {
  assert(particle.size() == _n_vis + _n_hid);
  _particles[k][i] = particle;
}

void Tracker::get_particle(int k, int i,
                           std::vector<double> *particle) {
  *particle = _particles[k][i];
}

void Tracker::run_particle(int k, int i, int n_samp) {
  double beta = _beta_vec[i];
  
  for (int iter=0; iter<n_samp; iter++) {
    for (int m=0; m<_n_hid; m++) {
      _particles[k][i][_n_vis+m] = beta * _c[m];
      for (int n=0; n<_n_vis; n++) {
        _particles[k][i][_n_vis+m] +=
            beta * _w[n*_n_hid+m] * _particles[k][i][n];
      }
      _particles[k][i][_n_vis+m] =
          bin_dist(sigmoid(_particles[k][i][_n_vis+m]));
    }
    for (int n=0; n<_n_vis; n++) {
      _particles[k][i][n] = beta * _b[n];
      for (int m=0; m<_n_hid; m++) {
        _particles[k][i][n] +=
            beta * _w[n*_n_hid+m] * _particles[k][i][_n_vis+m];
      }
      _particles[k][i][n] = bin_dist(sigmoid(_particles[k][i][n]));
    }
  }
}

void Tracker::run_all_particles(int n_samp) {
  for (int k=0; k<_n_batch; k++) {
    for (int i=0; i<_n_beta; i++) run_particle(k, i, n_samp);
  }
}

double Tracker::energy(int i_param, int k, int i_particle) {
  return energy(i_param, k, i_particle, _b, _c, _w);
}

double Tracker::energy(int i_param, int k, int i_particle,
                       const std::vector<double> &b,
                       const std::vector<double> &c,
                       const std::vector<double> &w) {
  double beta = _beta_vec[i_param];
  
  double e_val = 0.0;
  for (int n=0; n<_n_vis; n++)
    e_val -= beta * b[n] * _particles[k][i_particle][n];
  for (int m=0; m<_n_hid; m++)
    e_val -= beta * c[m] * _particles[k][i_particle][_n_vis+m];
  for (int n=0; n<_n_vis; n++) {
    for (int m=0; m<_n_hid; m++) {
      e_val -= beta *
          w[n*_n_hid+m] *
          _particles[k][i_particle][n] *
          _particles[k][i_particle][_n_vis+m];
    }
  }
  return e_val;
}

void Tracker::prob_swap(int k, int i1, int i2) {
  double swap_prob = exp(
    - energy(i1, k, i2) - energy(i2, k, i1)
    + energy(i1, k, i1) + energy(i2, k, i2)
  );
  
  if (swap_prob > uni_dist(gen))
    _particles[k][i1].swap(_particles[k][i2]);
}

void Tracker::prob_swap_all(bool odd) {
  for (int k=0; k<_n_batch; k++) {
    for (int i=odd; i<_n_beta-1; i+=2) {
      prob_swap(k, i, i+1);
    }
  }
}

double Tracker::log_part_func_det(int i) {
  assert(_n_vis < 32 || _n_hid < 32);
  double beta = _beta_vec[i];
  double z_val = 0.0;
  
  if (_n_vis < _n_hid) {
    std::vector<double> v(_n_vis, 0.0);
    for (int num_v=0; num_v<(1<<_n_vis); num_v++) {
      double beta_fe_val = 0.0;
      for (int n=0; n<_n_vis; n++) {
        v[n] = (double)((num_v >> n)&1);
        beta_fe_val -= beta * _b[n] * v[n];
      }
      for (int m=0; m<_n_hid; m++) {
        double h_act = beta * _c[m];
        for (int n=0; n<_n_vis; n++)
          h_act += beta * v[n] * _w[n*_n_hid+m];
        beta_fe_val -= log(1+exp(h_act));
      }
      z_val += exp(-beta_fe_val);
    }
  } else {
    std::vector<double> h(_n_hid, 0.0);
    for (int num_h=0; num_h<(1<<_n_hid); num_h++) {
      double beta_fe_val = 0.0;
      for (int m=0; m<_n_hid; m++) {
        h[m] = (double)((num_h >> m)&1);
        beta_fe_val -= beta * _c[m] * h[m];
      }
      for (int n=0; n<_n_vis; n++) {
        double v_act = beta * _b[n];
        for (int m=0; m<_n_hid; m++)
          v_act += beta * h[m] * _w[n*_n_hid+m];
        beta_fe_val -= log(1+exp(v_act));
      }
      z_val += exp(-beta_fe_val);
    }
  }
  return log(z_val);
}

void Tracker::bridge_sample(const std::vector<double> &o_delta_beta_in,
                            std::vector<double> *o_delta_beta_out,
                            std::vector<double> *sigma_delta_beta) {
  std::vector<double> u(_n_batch*(_n_beta-1), 0.0);
  std::vector<double> v(_n_batch*(_n_beta-1), 0.0);
  
  for (int k=0; k<_n_batch; k++) {
    for (int i=0; i<_n_beta-1; i++) {
      
      double s = exp(o_delta_beta_in[i]);
      
      // calculate u
      double q_star_num_u = exp(
          - energy(i, k, i) - energy(i+1, k, i) );
      double q_star_denom_u =
          s * exp(- energy(i, k, i)) + exp(- energy(i+1, k, i));
      double q_star_u = q_star_num_u / q_star_denom_u;
      u[k*(_n_beta-1)+i] = q_star_u / exp(- energy(i, k, i));
      
      // calculate v
      double q_star_num_v = exp(
          - energy(i, k, i+1) - energy(i+1, k, i+1) );
      double q_star_denom_v =
          s * exp(- energy(i, k, i+1)) + exp(- energy(i+1, k, i+1));
      double q_star_v = q_star_num_v / q_star_denom_v;
      v[k*(_n_beta-1)+i] = q_star_v / exp(- energy(i+1,k,i+1));
      
      // for numerical stability
      u[k*(_n_beta-1)+i] = clip(u[k*(_n_beta-1)+i]);
      v[k*(_n_beta-1)+i] = clip(v[k*(_n_beta-1)+i]);
    }
  }
  
  // calculate o_delta_beta_out and sigma_delta_beta
  (*o_delta_beta_out).resize(_n_beta-1);
  (*sigma_delta_beta).resize(_n_beta-1);
  for (int i=0; i<_n_beta-1; i++) {
    double u_accum = 0.0;
    double v_accum = 0.0;
    for (int k=0; k<_n_batch; k++) {
      u_accum += u[k*(_n_beta-1)+i];
      v_accum += v[k*(_n_beta-1)+i];
    }
    
    double u_mean = u_accum / _n_batch;
    double v_mean = v_accum / _n_batch;
    double u_squared_err = 0.0;
    double v_squared_err = 0.0;
    for (int k=0; k<_n_batch; k++) {
      u_squared_err +=
          (u[k*(_n_beta-1)+i] - u_mean) *
          (u[k*(_n_beta-1)+i] - u_mean);
      v_squared_err +=
          (v[k*(_n_beta-1)+i] - v_mean) *
          (v[k*(_n_beta-1)+i] - v_mean);
    }
    (*sigma_delta_beta)[i] =
        u_squared_err / u_accum / u_accum +
        v_squared_err / v_accum / v_accum;
    (*sigma_delta_beta)[i] *= _n_batch;
    
    (*o_delta_beta_out)[i] =
        log(u_accum) - log(v_accum) - (*sigma_delta_beta)[i] / 2.0;
  }
}

void Tracker::bridge_sample_mean_log(
    const std::vector<double> &o_delta_beta_in,
    std::vector<double> *o_delta_beta_out,
    std::vector<double> *sigma_delta_beta) {
  std::vector<double> log_u(_n_batch*(_n_beta-1), 0.0);
  std::vector<double> log_v(_n_batch*(_n_beta-1), 0.0);
  
  for (int k=0; k<_n_batch; k++) {
    for (int i=0; i<_n_beta-1; i++) {
      
      double s = exp(o_delta_beta_in[i]);
      
      // calculate u
      double q_star_num_u = exp(
          - energy(i, k, i) - energy(i+1, k, i) );
      double q_star_denom_u =
          s * exp(- energy(i, k, i)) + exp(- energy(i+1, k, i));
      double q_star_u = q_star_num_u / q_star_denom_u;
      log_u[k*(_n_beta-1)+i] = log(q_star_u) + energy(i, k, i);
      
      // calculate v
      double q_star_num_v = exp(
          - energy(i, k, i+1) - energy(i+1, k, i+1) );
      double q_star_denom_v =
          s * exp(- energy(i, k, i+1)) + exp(- energy(i+1, k, i+1));
      double q_star_v = q_star_num_v / q_star_denom_v;
      log_v[k*(_n_beta-1)+i] = log(q_star_v) + energy(i+1,k,i+1);
    }
  }
  
  // calculate o_delta_beta_out and sigma_delta_beta
  (*o_delta_beta_out).resize(_n_beta-1);
  (*sigma_delta_beta).resize(_n_beta-1);
  for (int i=0; i<_n_beta-1; i++) {
    double accum = 0.0;
    for (int k=0; k<_n_batch; k++)
      accum += clip(log_u[k*(_n_beta-1)+i] - log_v[k*(_n_beta-1)+i]);
        
    double mean = accum / _n_batch;
    double squared_err = 0.0;
    for (int k=0; k<_n_batch; k++) {
      squared_err +=
        (clip(log_u[k*(_n_beta-1)+i]-log_v[k*(_n_beta-1)+i]) - mean) *
        (clip(log_u[k*(_n_beta-1)+i]-log_v[k*(_n_beta-1)+i]) - mean);
    }
    (*sigma_delta_beta)[i] = squared_err / _n_batch;
    (*o_delta_beta_out)[i] = mean;
  }
}

void Tracker::import_sample(const std::vector<double> &b,
                            const std::vector<double> &c,
                            const std::vector<double> &w,
                            std::vector<double> *o_delta_t,
                            std::vector<double> *sigma_delta_t) {
  assert(b.size() == _n_vis);
  assert(c.size() == _n_hid);
  assert(w.size() == _n_vis*_n_hid);
  
  std::vector<double> omega(_n_batch*_n_beta, 0.0);
  
  for (int k=0; k<_n_batch; k++) {
    for (int i=0; i<_n_beta; i++) {
      omega[k*_n_beta+i] = exp(
          - energy(i, k, i, b, c, w) + energy(i, k, i)
      );
    }
  }
  
  (*o_delta_t).resize(_n_beta);
  (*sigma_delta_t).resize(_n_beta);
  for (int i=0; i<_n_beta; i++) {
    
    double omega_accum = 0.0;
    for (int k=0; k<_n_batch; k++)
      omega_accum += omega[k*_n_beta+i];
    double omega_mean = omega_accum / _n_batch;
    
    double omega_squared_err = 0.0;
    for (int k=0; k<_n_batch; k++) {
      omega_squared_err +=
          (omega[k*_n_beta+i] - omega_mean) *
          (omega[k*_n_beta+i] - omega_mean);
    }
    (*sigma_delta_t)[i] =
        omega_squared_err / omega_accum / omega_accum;
    (*sigma_delta_t)[i] *= _n_batch;
    
    (*o_delta_t)[i] = log(omega_mean) - (*sigma_delta_t)[i] / 2.0;
  }
}

void Tracker::import_sample_mean_log(
    const std::vector<double> &b,
    const std::vector<double> &c,
    const std::vector<double> &w,
    std::vector<double> *o_delta_t,
    std::vector<double> *sigma_delta_t) {
  assert(b.size() == _n_vis);
  assert(c.size() == _n_hid);
  assert(w.size() == _n_vis*_n_hid);
  
  std::vector<double> log_omega(_n_batch*_n_beta, 0.0);
  
  for (int k=0; k<_n_batch; k++) {
    for (int i=0; i<_n_beta; i++) {
      log_omega[k*_n_beta+i] =
          - energy(i, k, i, b, c, w) + energy(i, k, i);
    }
  }
  
  (*o_delta_t).resize(_n_beta);
  (*sigma_delta_t).resize(_n_beta);
  for (int i=0; i<_n_beta; i++) {
    
    double accum = 0.0;
    for (int k=0; k<_n_batch; k++)
      accum += log_omega[k*_n_beta+i];
    double mean = accum / _n_batch;
    
    double squared_err = 0.0;
    for (int k=0; k<_n_batch; k++) {
      squared_err +=
          (log_omega[k*_n_beta+i] - mean) *
          (log_omega[k*_n_beta+i] - mean);
    }
    (*sigma_delta_t)[i] = squared_err / _n_batch;
    (*o_delta_t)[i] = mean;
  }
}

void Tracker::init_mu_p(int n_samp, int n_bridge,
                        int n_swap, bool mean_log) {
  std::vector<double> o_delta_beta_in(_n_beta-1, 0.0);
  std::vector<double> o_delta_beta_out, sigma_delta_beta;
  run_all_particles(n_samp);
  for (int iter_swap=0; iter_swap<n_swap; iter_swap++) {
    prob_swap_all(false);
    prob_swap_all(true);
  }
  for (int iter=0; iter<n_bridge; iter++) {
    if (mean_log) {
      bridge_sample_mean_log(
          o_delta_beta_in, &o_delta_beta_out, &sigma_delta_beta);
    } else {
      bridge_sample(o_delta_beta_in,
          &o_delta_beta_out, &sigma_delta_beta);
    }
    o_delta_beta_in = o_delta_beta_out;
  }
  
  _mu.resize(_n_beta+1);
  std::fill(_mu.begin(), _mu.end(), 0.0);
  _mu[_n_beta-1] = (_n_vis+_n_hid) * log(2.0);
  _p.resize((_n_beta+1)*(_n_beta+1));
  std::fill(_p.begin(), _p.end(), 0.0);
  for (int i=_n_beta-2; i>=0; i--) {
    _mu[i] = _mu[i+1] - o_delta_beta_out[i];
    _p[i*(_n_beta+1)+i] =
        _p[(i+1)*(_n_beta+1)+(i+1)] + sigma_delta_beta[i];
  }
}

void Tracker::init_ais(int n_samp, int n_step) {
  _mu.resize(_n_beta+1);
  _p.resize((_n_beta+1)*(_n_beta+1));
  for (int i=0; i<(_n_beta+1)*(_n_beta+1); i++) _p[i] = 0.0;
  
  std::vector<double> b(_n_vis, 0.0);
  std::vector<double> c(_n_hid, 0.0);
  std::vector<double> w(_n_vis*_n_hid, 0.0);
  for (int i=0; i<_n_beta; i++) {
    for (int n=0; n<_n_vis; n++) b[n] = _beta_vec[i] * _b[n];
    for (int m=0; m<_n_hid; m++) c[m] = _beta_vec[i] * _c[m];
    for (int n=0; n<_n_vis*_n_hid; n++) w[n] = _beta_vec[i] * _w[n];
    
    lpf_ais(n_samp, n_step, b, c, w, &_mu[i], &_p[i*(_n_beta+1)*i]);
  }
}

void Tracker::update_mu_p(double sigma_z, double sigma_b,
                          std::vector<double> *o_delta_t,
                          std::vector<double> *sigma_delta_t,
                          std::vector<double> *o_delta_beta,
                          std::vector<double> *sigma_delta_beta) {
  // suppose _mu and _p are already initialized
  assert(_mu.size() == _n_beta+1);
  assert(_p.size() == (_n_beta+1)*(_n_beta+1));
  
  kalman_filter_impl(sigma_z, sigma_b,
                     o_delta_t, sigma_delta_t,
                     o_delta_beta, sigma_delta_beta,
                     &_mu, &_p, &_mu, &_p);
}

void Tracker::track(const std::vector<double> &b,
                    const std::vector<double> &c,
                    const std::vector<double> &w,
                    int n_samp_t, int n_samp_beta, int n_bridge,
                    double sigma_z, double sigma_b,
                    bool mean_log) {
  assert(b.size() == _n_vis);
  assert(c.size() == _n_hid);
  assert(w.size() == _n_vis * _n_hid);
  
  run_all_particles(n_samp_t);
  
  std::vector<double> o_delta_t, sigma_delta_t;
  if (mean_log) {
    import_sample_mean_log(b, c, w, &o_delta_t, &sigma_delta_t);
  } else {
    import_sample(b, c, w, &o_delta_t, &sigma_delta_t);
  }
  
  set_params(b, c, w);
  
  run_all_particles(n_samp_beta);
  prob_swap_all(false);
  prob_swap_all(true);
  
  std::vector<double> o_delta_beta_in(_n_beta-1, 0.0);
  for (int i=0; i<_n_beta-1; i++)
    o_delta_beta_in[i] = _mu[i+1] - _mu[i];
  std::vector<double> o_delta_beta, sigma_delta_beta;
  if (mean_log) {
    for (int iter=0; iter<n_bridge; iter++) {
      bridge_sample_mean_log(
          o_delta_beta_in, &o_delta_beta, &sigma_delta_beta);
    }
  } else {
    for (int iter=0; iter<n_bridge; iter++) {
        bridge_sample(
            o_delta_beta_in, &o_delta_beta, &sigma_delta_beta);
    }
  }
  
  update_mu_p(sigma_z, sigma_b,
              &o_delta_t, &sigma_delta_t,
              &o_delta_beta, &sigma_delta_beta);
}