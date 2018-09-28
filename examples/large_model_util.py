from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import six
import numpy as np
from scipy.special import expit

def _maybe_download(target_dir):
  target_path = os.path.join(target_dir, "mnist.pkl.gz")
  if not os.path.exists(target_dir):
    os.system(" ".join([
        "wget -P",
        target_dir,
        "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    ]))

def load_mnist(target_dir):
  _maybe_download(target_dir)
  target_path = os.path.join(target_dir, "mnist.pkl.gz")
  file_in = gzip.open(target_path, "rb")
  if six.PY2:
    import cPickle
    train, valid, test = cPickle.load(file_in)
  elif six.PY3:
    import pickle
    u = pickle._Unpickler(file_in)
    u.encoding = 'latin1'
    train, valid, test = u.load()
  images = np.vstack([train[0], valid[0], test[0]])
  n_vis = 784
  return images, n_vis

def calc_update(data, b, c, w, n_samp=1):
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

def _reconst(data, b, c, w):
  m_h = expit(np.dot(data, w) + c)
  return expit(np.dot(w, m_h.T).T + b)

def calc_xentropy(data, b, c, w):
  m_vis = _reconst(data, b, c, w)
  m_vis = np.clip(m_vis, 1e-8, 1 - 1e-8)
  xentropy = - np.mean(
    np.sum(
      data * np.log(m_vis) +
      (1-data) * np.log(1-m_vis),
    axis=1)
	)
  return xentropy
