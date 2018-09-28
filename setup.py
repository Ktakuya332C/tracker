from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

c_src_files = [
  "tracker/src/tracker.cc",
  "tracker/src/tracker_helper.cc",
  "tracker/src/math_helper.cc"
]
cython_src_files = [
  "tracker/wrap/tracker_wrap.pyx"
]
cython_module = Extension(
  "tracker.tracker_wrap",
  c_src_files + cython_src_files,
  include_dirs =["tracker/src", "third_party/eigen3"],
  language = "c++"
)

setup(
  name = "tracker",
  version = "0.0.1",
  description = "A library to track partition function of an RBM",
  long_description = (
    "An implementation of the paper"
    "G. Desjardins et al.(2011) On Tracking The Partition Function"),
  license = "MIT License",
  packages = ["tracker"],
  ext_modules = cythonize(cython_module),
  install_requires = [
    "cython==0.27.2",
    "numpy==1.14.0"
  ]
)