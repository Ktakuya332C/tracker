from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

import tracker

# Initialize RBM parameters
n_vis = 30
n_hid = 10
b = np.zeros(n_vis)
c = np.zeros(n_hid)
scale = np.sqrt(1.0 / (n_vis + n_hid))
w = np.random.uniform(low=-scale, high=scale, size=(n_vis, n_hid))

# Initialize PyTracker
n_beta = 100
n_batch = 30
tkr = tracker.PyTracker(n_beta, n_batch, n_vis, n_hid)
tkr.set_params(b, c, w)

# Initialize an initial estimate of partition function
tkr.init_mu_p() # Parallel tempering estimate
# tkr.init_ais() # AIS estimate

# Start main loop
n_epoch = 10
till_down = 5
mean_trk_list = list()
std_trk_list = list()
mean_ais_list = list()
std_ais_list = list()
mean_det_list = list()
for i in range(n_epoch):
  if i < till_down:
      w += 1e-4
  else:
      w -= 1e-4
  
  tkr.track(b, c, w)
  mean_trk, std_trk = tkr.get_estimates()
  mean_ais, std_ais = tracker.py_lpf_ais(b, c, w)
  mean_det = tracker.py_lpf_det(b, c, w)
  
  mean_trk_list.append(mean_trk)
  std_trk_list.append(std_trk)
  mean_ais_list.append(mean_ais)
  std_ais_list.append(std_ais)
  mean_det_list.append(mean_det)
  
  print("Finish", i, "th epoch")
  print("-- Log partition function calculated deterministically")
  print("---->", "mean:", mean_det)
  print("-- Log partition function calculated by tracker module")
  print("---->", "mean:", mean_trk, "std:", std_trk)
  print("-- Log partition function calculated using AIS")
  print("---->", "mean:", mean_ais, "std:", std_ais)

# Plot the simulation result calculated by tracker module
plt.errorbar(np.arange(n_epoch), mean_trk_list,
    yerr=std_trk_list, fmt="o", label="tracker module")
plt.plot(mean_det_list, label="deterministic calculation")
plt.title("tracker module")
plt.xlabel("epoch")
plt.ylabel("lpf")
plt.legend()
plt.savefig("small_model_track.png")
plt.close()

# Plot the simulation result calculated using AIS
plt.errorbar(np.arange(n_epoch), mean_ais_list,
    yerr=std_ais_list, fmt="o", label="AIS")
plt.plot(mean_det_list, label="deterministic calculation")
plt.title("AIS")
plt.xlabel("epoch")
plt.ylabel("lpf")
plt.legend()
plt.savefig("small_model_ais.png")
plt.close()
