from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

import tracker
from large_model_util import load_mnist, calc_update, calc_xentropy

# Load MNIST dataset
imgs, n_vis = load_mnist("data")

# Initialize RBM parameters
n_hid = 100
b = np.zeros(n_vis)
c = np.zeros(n_hid)
scale = np.sqrt(1.0 / (n_vis + n_hid))
w = np.random.normal(scale=scale, size=(n_vis, n_hid))

# Initialize PyTracker
n_beta = 10
n_batch = 10
tkr = tracker.PyTracker(n_beta, n_batch, n_vis, n_hid)
tkr.set_params(b, c, w)
tkr.init_mu_p()

# Start main loop
max_epoch = 101
train_batch = 20
learning_rate = 0.001
ais_epoch_list = list()
m_ais_list = list()
std_ais_list = list()
trk_epoch_list = list()
m_trk_list = list()
std_trk_list = list()
for epoch in range(max_epoch):
  
  # Extract a batch for this epoch
  img_idxs = np.random.choice(len(imgs), train_batch)
  batch = imgs[img_idxs]
  
  # Update RBM parameters
  ub, uc, uw = calc_update(batch, b, c, w)
  b += learning_rate * ub
  c += learning_rate * uc
  w += learning_rate * uw
  
  # Track the log partition function of the RBM
  tkr.track(b, c, w)
  m_trk, std_trk = tkr.get_estimates()
  trk_epoch_list.append(epoch)
  m_trk_list.append(m_trk)
  std_trk_list.append(std_trk)
  
  # Calculate the log partition function using AIS
  if epoch % 10 == 0:
    m_ais, std_ais = tracker.py_lpf_ais(b, c, w)
    ais_epoch_list.append(epoch)
    m_ais_list.append(m_ais)
    std_ais_list.append(std_ais)
    
    xent = calc_xentropy(batch, b, c, w)
    print("Finish", epoch, "th epoch")
    print("-- Cross entroy loss of the RBM")
    print("---->", xent)
    print("-- Log partition function calculated by tracker module")
    print("---->", "mean:", m_trk, "std:", std_trk)
    print("-- Log partition function calculated using AIS")
    print("---->", "mean:", m_ais, "std:", std_ais)

# Plot the simulation result
plt.errorbar(trk_epoch_list, m_trk_list,
    yerr=std_trk_list, fmt="o", label="tracker module")
plt.errorbar(ais_epoch_list, m_ais_list,
    yerr=std_ais_list, fmt="o", label="AIS")
plt.xlabel("epoch")
plt.ylabel("lpf")
plt.legend()
plt.savefig("large_model.png")
plt.close()
