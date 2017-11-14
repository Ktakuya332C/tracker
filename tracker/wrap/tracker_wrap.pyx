from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from libcpp.vector cimport vector

cdef extern from "tracker.h":
	cdef cppclass Tracker:
		Tracker(int n_beta, int n_batch, int n_vis, int n_hid) except +
		void set_params(vector[double] &b,
										vector[double] &c,
										vector[double] &w)
		void get_params(vector[double] *b,
										vector[double] *c,
										vector[double] *w)
		void get_estimates(double *lpf, double *std_lpf);
		void init_mu_p(int n_samp, int n_bridge, int n_swap, bint mean_log)
		void init_ais(int n_samp, int n_step)
		void track(vector[double] &b,
							 vector[double] &c,
							 vector[double] &w,
							 int n_samp_t, int n_samp_beta, int n_bridge,
							 double sigma_z, double sigma_b, bint mean_log)

cdef extern from "tracker_helper.h":
	void lpf_det(vector[double] b, vector[double] c, vector[double] w,
							 double *lpf, double *std_lpf)
	void lpf_ais(int n_samp, int n_step,
							 vector[double] b, vector[double] c, vector[double] w,
							 double *lpf, double *std_lpf)

def py_lpf_det(b, c, w):
	assert isinstance(b, np.ndarray), "b must be np.ndarray"
	assert isinstance(c, np.ndarray), "c must be np.ndarray"
	assert isinstance(w, np.ndarray), "w must be np.ndarray"
	assert w.shape == (len(b), len(c)), "shape of w is not appropriate"
	w = w.flatten()
	
	cdef double lpf
	cdef double std_lpf
	lpf_det(<const vector[double] &> b,
					<const vector[double] &> c,
					<const vector[double] &> w,
					<double*> &lpf, <double*> &std_lpf)
	return lpf

def py_lpf_ais(b, c, w, n_samp=10, n_step=10000):
	assert isinstance(b, np.ndarray), "b must be np.ndarray"
	assert isinstance(c, np.ndarray), "c must be np.ndarray"
	assert isinstance(w, np.ndarray), "w must be np.ndarray"
	assert w.shape == (len(b), len(c)), "shape of w is not appropriate"
	w = w.flatten()
	
	cdef double lpf
	cdef double std_lpf
	lpf_ais(<int> n_samp, <int> n_step,
					<const vector[double] &> b,
					<const vector[double] &> c,
					<const vector[double] &> w,
					<double*> &lpf, <double*> &std_lpf)
	return lpf, std_lpf

cdef class PyTracker:
	cdef Tracker *c_impl
	cdef int n_beta, n_batch, n_vis, n_hid
	
	def __cinit__(self, int n_beta, int n_batch, int n_vis, int n_hid):
		self.c_impl = new Tracker(n_beta, n_batch, n_vis, n_hid)
	
	def __dealloc__(self):
		del self.c_impl
	
	def __init__(self, n_beta, n_batch, n_vis, n_hid):
		self.n_beta = n_beta
		self.n_batch = n_batch
		self.n_vis = n_vis
		self.n_hid = n_hid
	
	def set_params(self, b, c, w):
		assert isinstance(b, np.ndarray), "b must be np.ndarray"
		assert isinstance(c, np.ndarray), "c must be np.ndarray"
		assert isinstance(w, np.ndarray), "w must be np.ndarray"
		assert len(b) == self.n_vis, "length of b is not appropriate"
		assert len(c) == self.n_hid, "length of c is not appropriate"
		assert w.shape == (self.n_vis, self.n_hid), (
				"shape of w is not appropriate")
		w = w.flatten()
		
		self.c_impl.set_params(<const vector[double]&> b,
													 <const vector[double]&> c,
													 <const vector[double]&> w)
	
	def get_params(self):
		cdef vector[double] b
		cdef vector[double] c
		cdef vector[double] w
		self.c_impl.get_params(<vector[double]*> &b,
													 <vector[double]*> &c,
													 <vector[double]*> &w)
		return b, c, w
	
	def get_estimates(self):
		cdef double lpf;
		cdef double std_lpf;
		self.c_impl.get_estimates(<double*> &lpf, <double*> &std_lpf);
		return lpf, std_lpf
	
	def init_mu_p(self, n_samp=20, n_bridge=20,
								n_swap=10, mean_log=True):
		self.c_impl.init_mu_p(<int> n_samp, <int> n_bridge,
													<int> n_swap, <bint> mean_log)
	
	def init_ais(self, n_samp=10, n_step=10000):
		self.c_impl.init_ais(<int> n_samp, <int> n_step)
	
	def track(self, b, c, w,
						n_samp_t=50, n_samp_beta=50, n_bridge=10,
						sigma_z=1e6, sigma_b=1e-3,
						mean_log = True):
		assert isinstance(b, np.ndarray), "b must be np.ndarray"
		assert isinstance(c, np.ndarray), "c must be np.ndarray"
		assert isinstance(w, np.ndarray), "w must be np.ndarray"
		assert len(b) == self.n_vis, "length of b is not appropriate"
		assert len(c) == self.n_hid, "length of c is not appropriate"
		assert w.shape == (self.n_vis, self.n_hid), (
				"shape of w is not appropriate")
		w = w.flatten()
		
		self.c_impl.track(<const vector[double]&> b,
											<const vector[double]&> c,
											<const vector[double]&> w,
											<int> n_samp_t, <int> n_samp_beta,
											<int> n_bridge, <double> sigma_z,
											<double> sigma_b, <bint> mean_log)
		