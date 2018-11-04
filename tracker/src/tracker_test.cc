#include <cmath>
#include <algorithm>

#include "test_helper.h"
#include "math_helper.h"
#include "tracker.h"

void test_of_test(void) {
  assert_stoc_eq(1.0, 1.0);
  assert_det_eq(1.0, 1.0);
}

void test_tracker_init(void) {
  int n_beta=5, n_batch=30, n_vis=30, n_hid=20;
  
  std::vector<double> b_in(n_vis, 0.0);
  std::vector<double> c_in(n_hid, 0.0);
  std::vector<double> w_in(n_vis*n_hid, 0.0);
  for (int i=0; i<n_vis; i++) b_in[i] = sin(i);
  for (int i=0; i<n_vis; i++) c_in[i] = cos(i);
  for (int i=0; i<n_vis*n_hid; i++) w_in[i] = sin(i);
  
  Tracker tracker = Tracker(n_beta, n_batch, b_in, c_in, w_in);
  
  std::vector<double> b_out(n_vis, 0.0);
  std::vector<double> c_out(n_hid, 0.0);
  std::vector<double> w_out(n_vis*n_hid, 0.0);
  
  tracker.get_params(&b_out, &c_out, &w_out);
  
  for (int i=0; i<n_vis; i++) assert_det_eq(b_in[i], b_out[i]);
  for (int i=0; i<n_hid; i++) assert_det_eq(c_in[i], c_out[i]);
  for (int i=0; i<n_vis*n_hid; i++) assert_det_eq(w_in[i], w_out[i]);
  
  std::vector<double> particle;
  for (int k=0; k<n_batch; k++) {
    for (int i=0; i<n_beta; i++) {
      tracker.get_particle(k, i, &particle);
      for (int n=0; n<n_vis; n++)
        assert_bool(particle[n] == 1.0 || particle[n] == 0.0);
    }
  }
  
  std::vector<double> beta_vec;
  tracker.get_beta_vec(&beta_vec);
  assert_det_eq(beta_vec[0], 1.0);
  assert_det_eq(beta_vec[n_beta-1], 0.0);
}

void test_tracker_simple_init(void) {
  int n_beta=10, n_batch=15, n_vis=20, n_hid=30;
  
  Tracker tracker = Tracker(n_beta, n_batch, n_vis, n_hid);
  
  std::vector<double> b_in(n_vis, 0.0);
  std::vector<double> c_in(n_hid, 0.0);
  std::vector<double> w_in(n_vis*n_hid, 0.0);
  
  tracker.get_params(&b_in, &c_in, &w_in);
  
  for (int i=0; i<n_vis; i++) assert_det_eq(b_in[i], 0.0);
  for (int i=0; i<n_hid; i++) assert_det_eq(c_in[i], 0.0);
  for (int i=0; i<n_vis*n_hid; i++) assert_det_eq(w_in[i], 0.0);
}

void test_tracker_run_particle(void) {
  int n_beta=10, n_batch=15, n_vis=20, n_hid=30;
  
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, 0.0);
  std::vector<double> w(n_vis*n_hid, 0.0);
  b[0] = 100.0;
  b[1] = -100.0;
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  tracker.run_particle(0, 0, 20);
  
  std::vector<double> particle;
  tracker.get_particle(0, 0, &particle);
  assert_stoc_eq(particle[0], 1.0);
  assert_stoc_eq(particle[1], 0.0);
  for (int n=2; n<n_vis; n++)
    assert_bool(particle[n] == 1.0 || particle[n] == 0.0);
  
  b[0] = 0.0;
  b[1] = 0.0;
  std::fill(w.begin(), w.end(), 100.0);
  tracker.set_params(b, c, w);
  tracker.run_particle(0, 0, 20);
  tracker.get_particle(0, 0, &particle);
  assert_stoc_eq(particle[0], 1.0);
  assert_stoc_eq(particle[1], 1.0);
}

void test_tracker_energy(void) {
  int n_beta=5, n_batch=1, n_vis=4, n_hid=3;
  
  std::vector<double> b(n_vis, 0.0);
  for (int n=0; n<n_vis; n++) b[n] = n * 0.1;
  std::vector<double> c(n_hid, 0.0);
  for (int m=0; m<n_hid; m++) c[m] = m * 0.1;
  std::vector<double> w(n_vis*n_hid, 0.0);
  for (int l=0; l<n_vis*n_hid; l++) w[l] = l * 0.1;
  
  Tracker tracker = Tracker(n_beta, n_batch, b, c, w);
  
  std::vector<double> particle(n_vis+n_hid, 0.0);
  tracker.set_particle(0, 0, particle);
  assert_det_eq(tracker.energy(0, 0, 0), 0.0);
  
  std::fill(particle.begin(), particle.end(), 1.0);
  tracker.set_particle(0, 0, particle);
  assert_det_eq(tracker.energy(n_beta-1, 0, 0), 0.0);
  assert_det_eq(tracker.energy(0, 0, 0), -7.5);
}

void test_tracker_prob_swap(void) {
  int n_beta=7, n_batch=1, n_vis=10, n_hid=5;
  
  std::vector<double> b(n_vis, 1.0);
  std::vector<double> c(n_hid, 0.0);
  std::vector<double> w(n_vis*n_hid, 0.0);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  std::vector<double> particle(n_vis+n_hid, 0.0);
  tracker.set_particle(0, 0, particle);
  std::fill(particle.begin(), particle.end(), 1.0);
  tracker.set_particle(0, n_beta-1, particle);
  tracker.prob_swap(0, 0, n_beta-1);
  tracker.get_particle(0, 0, &particle);
  for (int n=0; n<n_vis+n_hid; n++)
    assert_det_eq(particle[n], 1.0);
}

void test_tracker_log_part_func_det(void) {
  int n_beta=2, n_batch=1, n_vis1=3, n_hid1=2, n_vis2=2, n_hid2=3;
  
  std::vector<double> b1(n_vis1, 0.0);
  std::vector<double> c1(n_hid1, 0.0);
  std::vector<double> w1(n_vis1*n_hid1, 0.0);
  Tracker tracker1 = Tracker(n_beta, n_batch,  b1, c1, w1);

  std::vector<double> b2(n_vis2, 0.0);
  std::vector<double> c2(n_hid2, 0.0);
  std::vector<double> w2(n_vis2*n_hid2, 0.0);
  Tracker tracker2 = Tracker(n_beta, n_batch,  b2, c2, w2);
  
  assert_det_eq(tracker1.log_part_func_det(0), 3.4657359027);
  assert_det_eq(tracker2.log_part_func_det(0), 3.4657359027);
  std::fill(b1.begin(), b1.end(), 1.0);
  std::fill(b2.begin(), b2.end(), 1.0);
  tracker1.set_params(b1, c1, w1);
  tracker2.set_params(b2, c2, w2);
  assert_det_eq(tracker1.log_part_func_det(1), 3.4657359027);
  assert_det_eq(tracker1.log_part_func_det(0), 5.3260794236);
  assert_det_eq(tracker2.log_part_func_det(1), 3.4657359027);
  assert_det_eq(tracker2.log_part_func_det(0), 4.7059649167);
}

void test_tracker_bridge_sample1(void) {
  int n_beta=10, n_batch=10, n_vis=15, n_hid=10;
  
  Tracker tracker = Tracker(n_beta, n_batch,  n_vis, n_hid);
  
  std::vector<double> o_delta_beta_in(n_beta-1, 0.0);
  std::vector<double> o_delta_beta_out, sigma_delta_beta;
  for (int iter=0; iter<20; iter++) {
    tracker.bridge_sample(o_delta_beta_in,
                          &o_delta_beta_out, &sigma_delta_beta);
    o_delta_beta_in = o_delta_beta_out;
  }
  for (int i=0; i<n_beta-1; i++)
    assert_stoc_eq(o_delta_beta_out[i], 0.0);
}

void test_tracker_bridge_sample2(void) {
  int n_beta=30, n_batch=30, n_vis=300, n_hid=10, n_swap=1;
  
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, -0.01);
  std::vector<double> w(n_vis*n_hid, 0.02);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  std::vector<double> o_delta_beta_in(n_beta-1, 0.0);
  std::vector<double> o_delta_beta_out, sigma_delta_beta;
  for (int iter=0; iter<10; iter++) {
    tracker.run_all_particles(20);
    for (int iter_swap=0; iter_swap<n_swap; iter_swap++) {
      tracker.prob_swap_all(false);
      tracker.prob_swap_all(true);
    }
    tracker.bridge_sample(o_delta_beta_in,
                          &o_delta_beta_out, &sigma_delta_beta);
    o_delta_beta_in = o_delta_beta_out;
  }
  
  double log_z = (n_hid+n_vis) * log(2.0);
  double var_log_z = 0.0;
  for (int i=n_beta-1; i>=0; i--) {
    log_z -= o_delta_beta_out[i];
    var_log_z += sigma_delta_beta[i];
    
    /*
    std::cout << "estimate " << log_z << std::endl;
    std::cout << "std est  " << sqrt(var_log_z) << std::endl;
    std::cout << "det      " <<
        tracker.log_part_func_det(i) << std::endl;
    */
  }
  double log_z_det = tracker.log_part_func_det(0);
  assert_bool(log_z - sqrt(var_log_z) <= log_z_det);
  assert_bool(log_z_det <= log_z + sqrt(var_log_z));
  
  /*
  std::cout << "log_z determnistic " << log_z_det << std::endl;
  std::cout << "log_z estimate " << log_z << std::endl;
  std::cout << "log_z std " << sqrt(var_log_z) << std::endl;
  */
}

void test_tracker_bridge_sample_mean_log(void) {
  int n_beta=30, n_batch=30, n_vis=300, n_hid=10, n_swap=1;
  
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, -0.01);
  std::vector<double> w(n_vis*n_hid, 0.02);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  std::vector<double> o_delta_beta_in(n_beta-1, 0.0);
  std::vector<double> o_delta_beta_out, sigma_delta_beta;
  for (int iter=0; iter<10; iter++) {
    tracker.run_all_particles(20);
    for (int iter_swap=0; iter_swap<n_swap; iter_swap++) {
      tracker.prob_swap_all(false);
      tracker.prob_swap_all(true);
    }
    tracker.bridge_sample_mean_log(
        o_delta_beta_in, &o_delta_beta_out, &sigma_delta_beta);
    o_delta_beta_in = o_delta_beta_out;
  }
  
  double log_z = (n_hid+n_vis) * log(2.0);
  double var_log_z = 0.0;
  for (int i=n_beta-1; i>=0; i--) {
    log_z -= o_delta_beta_out[i];
    var_log_z += sigma_delta_beta[i];
    
    /*
    std::cout << "estimate " << log_z << std::endl;
    std::cout << "std est  " << sqrt(var_log_z) << std::endl;
    std::cout << "det      " <<
        tracker.log_part_func_det(i) << std::endl;
    */
  }
  double log_z_det = tracker.log_part_func_det(0);
  assert_bool(log_z - sqrt(var_log_z) <= log_z_det);
  assert_bool(log_z_det <= log_z + sqrt(var_log_z));
  
  /*
  std::cout << "log_z determnistic " << log_z_det << std::endl;
  std::cout << "log_z estimate " << log_z << std::endl;
  std::cout << "log_z std " << sqrt(var_log_z) << std::endl;
  */
}

void test_tracker_import_sample1(void) {
  int n_beta=10, n_batch=100, n_vis=15, n_hid=10;
  
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, 0.0);
  std::vector<double> w(n_vis*n_hid, 0.0);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  std::vector<double> o_delta_t, sigma_delta_t;
  tracker.import_sample(b, c, w, &o_delta_t, &sigma_delta_t);
  
  assert_det_eq(o_delta_t.size(), n_beta);
  for (int i=0; i<n_beta; i++)
    assert_stoc_eq(o_delta_t[i], 0.0);
}

void test_tracker_import_sample2(void) {
  int n_beta=2, n_batch=100, n_vis=15, n_hid=10;
  
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, 0.0);
  std::vector<double> w(n_vis*n_hid, 0.0);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  std::fill(w.begin(), w.end(), 0.01);
  std::vector<double> o_delta_t, sigma_delta_t;
  tracker.import_sample(b, c, w, &o_delta_t, &sigma_delta_t);
  
  assert_bool(0.0-2*sqrt(sigma_delta_t[1]) <= o_delta_t[1]);
  assert_bool(o_delta_t[1] <= 0.0+2*sqrt(sigma_delta_t[1]));
  tracker.set_params(b, c, w);
  double res_det = tracker.log_part_func_det(0)-(n_vis+n_hid)*log(2.0);
  
  assert_bool(o_delta_t[0] - 2*sqrt(sigma_delta_t[0]) <= res_det);
  assert_bool(res_det <= o_delta_t[0] + 2*sqrt(sigma_delta_t[0]));
}

void test_tracker_import_sample_mean_log(void) {
  int n_beta=2, n_batch=100, n_vis=15, n_hid=10;
  
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, 0.0);
  std::vector<double> w(n_vis*n_hid, 0.0);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  std::fill(w.begin(), w.end(), 0.01);
  std::vector<double> o_delta_t, sigma_delta_t;
  tracker.import_sample_mean_log(b, c, w, &o_delta_t, &sigma_delta_t);
  
  assert_bool(0.0-2*sqrt(sigma_delta_t[1]) <= o_delta_t[1]);
  assert_bool(o_delta_t[1] <= 0.0+2*sqrt(sigma_delta_t[1]));
  tracker.set_params(b, c, w);
  double res_det = tracker.log_part_func_det(0)-(n_vis+n_hid)*log(2.0);
  
  assert_bool(o_delta_t[0] - 2*sqrt(sigma_delta_t[0]) <= res_det);
  assert_bool(res_det <= o_delta_t[0] + 2*sqrt(sigma_delta_t[0]));
}

void test_tracker_init_mu_p(void) {
  int n_beta=10, n_batch=10, n_vis=20, n_hid=10;
  
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, 0.0);
  std::vector<double> w(n_vis*n_hid, 0.0);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  tracker.init_mu_p(20, 10, 10, false);
  
  // TODO: write tests
}

void test_tracker_update_mu_p(void) {
  int n_beta=10, n_batch=10, n_vis=30, n_hid=20;
  
  // initialize tracker
  std::vector<double> b(n_vis, 0.0);
  std::vector<double> c(n_hid, 0.0);
  std::vector<double> w(n_vis*n_hid, 0.01);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  // initialize mu and p
  tracker.init_mu_p(20, 10, 10, false);
  
  // kalman filter
  std::vector<double> mu, p;
  std::fill(w.begin(), w.end(), 0.02);
  for (int i=0; i<10; i++) {
    tracker.get_mu(&mu);
    tracker.get_p(&p);
    // TODO: write tests
  }
}

void test_tracker_track(void) {
  int n_beta=30, n_batch=30, n_vis=20, n_hid=10;
  
  std::vector<double> b(n_vis, -0.01);
  std::vector<double> c(n_hid, 0.01);
  std::vector<double> w(n_vis*n_hid, 0.01);
  Tracker tracker = Tracker(n_beta, n_batch,  b, c, w);
  
  double lpf, std_lpf, det_lpf;
  tracker.init_mu_p(50, 50, 10, true);
  for (int iter=0; iter<9; iter++) {
    std::fill(w.begin(), w.end(), iter*1e-5);
    tracker.track(b, c, w, 20, 20, 10, 1e4, 1e-3, true);
    tracker.get_estimates(&lpf, &std_lpf);
    if (iter % 3 == 0) {
      det_lpf = tracker.log_part_func_det(0);
      assert_bool(std::isfinite(lpf));
      assert_bool(std::isfinite(std_lpf));
      assert_bool(std::isfinite(det_lpf));
      assert_bool(lpf - 2*std_lpf <= det_lpf);
      assert_bool(det_lpf <= lpf + 2*std_lpf);
      
      /*
      std::cout << "Det  " << det_lpf << std::endl;
      std::cout << "Stoc " << lpf << " std " << std_lpf << std::endl;
      */
    }
  }
}

int main() {
  test_of_test();
  test_tracker_init();
  test_tracker_simple_init();
  test_tracker_run_particle();
  test_tracker_energy();
  test_tracker_prob_swap();
  test_tracker_log_part_func_det();
  test_tracker_bridge_sample1();
  test_tracker_bridge_sample2();
  test_tracker_bridge_sample_mean_log();
  test_tracker_import_sample1();
  test_tracker_import_sample2();
  test_tracker_import_sample_mean_log();
  test_tracker_init_mu_p();
  test_tracker_update_mu_p();
  test_tracker_track();
}