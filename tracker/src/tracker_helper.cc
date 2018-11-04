#include "tracker_helper.h"
#include "Eigen/Dense"
#include "math_helper.h"
using namespace Eigen;

void kalman_filter_impl(double sigma_z, double sigma_b,
                        std::vector<double> *o_delta_t_in,
                        std::vector<double> *sigma_delta_t_in,
                        std::vector<double> *o_delta_beta_in,
                        std::vector<double> *sigma_delta_beta_in,
                        std::vector<double> *mu_in,
                        std::vector<double> *p_in,
                        std::vector<double> *mu_out,
                        std::vector<double> *p_out) {
  // algorithm based on the paper G. Desjardins et al. (2011), NIPS
  // update equations are corrected based on PRML
  
  int M = o_delta_t_in->size();
  
  MatrixXd c(M, 2*M+2);
  c << -MatrixXd::Identity(M, M), VectorXd::Zero(M),
      MatrixXd::Identity(M, M), VectorXd::Zero(M);
  c(0, 2*M+1) = 1.0;
  MatrixXd h = MatrixXd::Zero(M-1, M+1);
  h.block(0, 0, M-1, M-1) -= MatrixXd::Identity(M-1, M-1);
  h.block(0, 1, M-1, M-1) += MatrixXd::Identity(M-1, M-1);
  
  // construct joint distribution of zeta_{t-1} and zeta_{t}
  VectorXd mu = Map<VectorXd>(mu_in->data(), M+1);
  VectorXd eta(2*M+2);
  eta << mu, mu;
  
  VectorXd sigma_zeta_diag = VectorXd::Ones(M+1);
  sigma_zeta_diag *= sigma_z * sigma_z;
  sigma_zeta_diag[M] = sigma_b * sigma_b;
  MatrixXd sigma_zeta = sigma_zeta_diag.asDiagonal();
  
  MatrixXd p = Map<MatrixXd>(p_in->data(), M+1, M+1);
  MatrixXd v(2*M+2, 2*M+2);
  v << p, p, p, p+sigma_zeta;
  
  // calculate posterior of zeta_{t-1} and zeta_{t}
  // conditioned by o_delta_t_in
  VectorXd sigma_delta_t_vec =
      Map<VectorXd>(sigma_delta_t_in->data(), M);
  MatrixXd sigma_delta_t = sigma_delta_t_vec.asDiagonal();
  VectorXd o_delta_t = Map<VectorXd>(o_delta_t_in->data(), M);
  
  MatrixXd m_t = v * c.transpose() *
      (sigma_delta_t + c * v * c.transpose()).inverse();
  assert(m_t.array().isFinite().all());
  eta += m_t * (o_delta_t - c * eta);
  v -= m_t * c * v;
  
  // maginalise over zeta_{t-1}
  mu = eta.tail(M+1);
  p = v.block(M+1, M+1, M+1, M+1);
  
  // calculate posterior of zeta_t conditioned by o_delta_beta_in
  VectorXd sigma_delta_beta_vec =
      Map<VectorXd>(sigma_delta_beta_in->data(), M-1);
  MatrixXd sigma_delta_beta = sigma_delta_beta_vec.asDiagonal();
  VectorXd o_delta_beta =
      Map<VectorXd>(o_delta_beta_in->data(), M-1);
  
  MatrixXd m_beta = p * h.transpose() *
      (sigma_delta_beta + h * p * h.transpose()).inverse();
  assert(m_beta.array().isFinite().all());
  mu += m_beta * (o_delta_beta - h * mu);
  p -= m_beta * h * p;
  
  // Output
  (*mu_out).resize(M+1);
  (*p_out).resize((M+1)*(M+1));
  Map<VectorXd>(mu_out->data(), M+1) = mu;
  Map<MatrixXd>(p_out->data(), M+1, M+1) = p;
}

double fe(const std::vector<double> &b,
          const std::vector<double> &c,
          const std::vector<double> &w,
          const std::vector<double> &v,
          double beta) {
  int n_vis = b.size();
  int n_hid = c.size();
  assert(w.size() == n_vis * n_hid);
  assert(v.size() == n_vis);
  
  double val = 0.0;
  for (int n=0; n<n_vis; n++)
    val -= b[n] * v[n];
  double hid_inpt;
  for (int m=0; m<n_hid; m++) {
    hid_inpt = 0.0;
    for (int n=0; n<n_vis; n++)
      hid_inpt += v[n] * w[n*n_hid+m];
    val -= log(1.0 + exp(beta * (c[m] + hid_inpt)));
  }
  return val;
}

void pv_given_h(double beta,
                const std::vector<double> &b,
                const std::vector<double> &c,
                const std::vector<double> &w,
                const std::vector<double> &h,
                std::vector<double> *v) {
  int n_vis = b.size();
  int n_hid = c.size();
  
  for (int n=0; n<n_vis; n++) {
    (*v)[n] = b[n];
    for (int m=0; m<n_hid; m++)
      (*v)[n] += beta * h[m] * w[n*n_hid+m];
    (*v)[n] = bin_dist(sigmoid((*v)[n]));
  }
}

void ph_given_v(double beta,
                const std::vector<double> &b,
                const std::vector<double> &c,
                const std::vector<double> &w,
                const std::vector<double> &v,
                std::vector<double> *h) {
  int n_vis = b.size();
  int n_hid = c.size();
  
  for (int m=0; m<n_hid; m++) {
    (*h)[m] = beta * c[m];
    for (int n=0; n<n_vis; n++)
      (*h)[m] += beta * v[n] * w[n*n_hid+m];
    (*h)[m] = bin_dist(sigmoid((*h)[m]));
  }
}

void lpf_ais(int n_samp, int n_step,
             const std::vector<double> &b,
             const std::vector<double> &c,
             const std::vector<double> &w,
             double *lpf, double *std_lpf) {
  int n_vis = b.size();
  int n_hid = c.size();
  assert(w.size() == n_vis * n_hid);
  std::vector<double> lpfs(n_samp, 0.0);
  
  int k;
  for (k=0; k<n_samp; k++) {
    double beta;
    std::vector<double> v(n_vis, 0.0);
    std::vector<double> h(n_hid, 0.0);
    
    for (int n=0; n<n_vis; n++)
      v[n] = bin_dist(sigmoid(b[n]));
    
    beta = 0.0;
    lpfs[k] += fe(b, c, w, v, beta);
    beta = 1.0 / (double)n_step;
    lpfs[k] -= fe(b, c, w, v, beta);
    
    for (int i=1; i<n_step; i++) {
      beta = (double)i / (double)n_step;
      ph_given_v(beta, b, c, w, v, &h);
      pv_given_h(beta, b, c, w, h, &v);
      
      lpfs[k] += fe(b, c, w, v, beta);
      beta = (double)(i+1) / (double)n_step;
      lpfs[k] -= fe(b, c, w, v, beta);
    }
  }
  
  *lpf = 0.0;
  for (int k=0; k<n_samp; k++)
    *lpf += lpfs[k];
  *lpf /= (double)n_samp;
  
  *std_lpf = 0.0;
  for (int k=0; k<n_samp; k++)
    *std_lpf += (lpfs[k] - *lpf) * (lpfs[k] - *lpf);
  *std_lpf = sqrt(*std_lpf / (double)(n_samp*(n_samp-1)));
  
  *lpf += (double)n_hid * log(2.0);
  for (int n=0; n<n_vis; n++)
    *lpf += log(1.0 + exp(b[n]));
}

void lpf_det(const std::vector<double> &b,
             const std::vector<double> &c,
             const std::vector<double> &w,
             double *lpf, double *std_lpf) {
  int n_vis = b.size();
  int n_hid = c.size();
  assert(w.size() == n_vis * n_hid);
  assert(n_vis < 32 || n_hid < 32);
  
  *lpf = 0.0;
  *std_lpf = 0.0;
  
  if (n_vis < n_hid) {
    std::vector<double> v(n_vis, 0.0);
    for (int num_v=0; num_v < (1<<n_vis); num_v++) {
      double beta_fe_val = 0.0;
      for (int n=0; n<n_vis; n++) {
        v[n] = (double)((num_v>>n)&1);
        beta_fe_val -= b[n] * v[n];
      }
      for (int m=0; m<n_hid; m++) {
        double h_act = c[m];
        for (int n=0; n<n_vis; n++)
          h_act += v[n] * w[n*n_hid+m];
        beta_fe_val -= log(1.0 + exp(h_act));
      }
      *lpf += exp(-beta_fe_val);
    }
  } else {
    std::vector<double> h(n_hid, 0.0);
    for (int num_h=0; num_h < (1<<n_hid); num_h++) {
      double beta_fe_val = 0.0;
      for (int m=0; m<n_hid; m++) {
        h[m] = (double)((num_h>>m)&1);
        beta_fe_val -= c[m] * h[m];
      }
      for (int n=0; n<n_vis; n++) {
        double v_act = b[n];
        for (int m=0; m<n_hid; m++)
          v_act += h[m] * w[n*n_hid+m];
        beta_fe_val -= log(1.0 + exp(v_act));
      }
      *lpf += exp(-beta_fe_val);
    }
  }
  *lpf = log(*lpf);
}
