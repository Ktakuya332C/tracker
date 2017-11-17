from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import six
import numpy as np
from scipy.special import expit

def load_params(path, name_base):
  b = np.load(os.path.join(path, "b"+name_base+".npy"))
  c = np.load(os.path.join(path, "c"+name_base+".npy"))
  w = np.load(os.path.join(path, "w"+name_base+".npy"))
  return b, c, w

def save_params(b, c, w, name_base, log_dir):
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  
  np.save(os.path.join(log_dir, "b"+name_base), b)
  np.save(os.path.join(log_dir, "c"+name_base), c)
  np.save(os.path.join(log_dir, "w"+name_base), w)

def load_mnist(path):
  file_in = gzip.open(path, "rb")
  if six.PY2:
    import cPickle
    train, valid, test = cPickle.load(file_in)
  elif six.PY3:
    import pickle
    u = pickle._Unpickler(file_in)
    u.encoding = 'latin1'
    train, valid, test = u.load()
  images = np.vstack([train[0], valid[0], test[0]])
  return images, 784

def load_data(path):
  data = np.load(path)
  data = data.astype(np.float)
  assert len(data.shape) == 2, "The shape of the input data is not "
  return data, data.shape[1]

def calc_update_stoc(data, b, c, w, n_samp=1):
  m_vis = data
  m_hid = expit(np.dot(data, w) + c)
  s_vis = m_vis
  for i in range(n_samp):
    sm_hid = expit(np.dot(s_vis, w) + c)
    s_hid = np.random.binomial(1, sm_hid)
    sm_vis = expit(np.dot(s_hid, w.T) + b)
    s_vis = np.random.binomial(1, sm_vis)
  ub = np.mean(m_vis - s_vis, axis=0)
  uc = np.mean(m_hid - s_hid, axis=0)
  uw = (np.dot(m_vis.T, m_hid) - np.dot(s_vis.T, s_hid)) / len(data)
  return ub, uc, uw

def calc_update_det(data, b, c, w):
  m_vis = data
  m_hid = expit(np.dot(data, w) + c)
  m_b = np.mean(data, axis=0)
  m_c = np.mean(m_hid, axis=0)
  m_w = np.dot(m_vis.T, m_hid) / len(data)
  
  s_b = np.zeros(b.shape)
  s_c = np.zeros(c.shape)
  s_w = np.zeros(w.shape)
  z = 0
  for bin_vec in itertools.product([0, 1], repeat=len(b)):
    v_act = np.array(bin_vec)
    h_act = expit(c + np.dot(v_act, w))
    prob_weight = np.exp(-free_energy(v_act.reshape(1, len(b)), b, c, w))
    z += prob_weight
    s_b += prob_weight* v_act
    s_c += prob_weight * h_act
    s_w += prob_weight * np.outer(v_act, h_act)
  s_b = s_b / z
  s_c = s_c / z
  s_w = s_w / z
  
  ub = m_b - s_b
  uc = m_c - s_c
  uw = m_w - s_w
  return ub, uc, uw

def save_reconst(data, b, c, w, name_base, log_dir):
  reconst_images = reconst(data, b, c, w)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  np.save(os.path.join(log_dir, "reconst"+name_base), reconst_images)

def reconst(data, b, c, w):
  m_hid = expit(np.dot(data, w) + c)
  m_vis = expit(np.dot(m_hid, w.T) + b)
  return m_vis

def calc_xentropy(data, b, c, w):
  m_vis = reconst(data, b, c, w)
  m_vis = np.clip(m_vis, 1e-8, 1 - 1e-8)
  xentropy = (- np.mean(np.sum(data * np.log(m_vis) +
	    (1-data)*np.log(1-m_vis), axis=1)))
  return xentropy