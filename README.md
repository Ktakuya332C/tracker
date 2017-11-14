Partition Function Tracker
====
An implementation of the paper "On Tracking The Partition Function" in python and c++

## Description
An implementation of a partition function tracking method for RBM described in ref[1]. Since direct calculation of partition function of RBM is infeasible in most cases, the method uses Kalman filter on many easily available cues from RBM to estimate partition function. Both mean estimate of partition function and its standard deviation estimates should be available. 

Also repository contains AIS (Annealed importance sampling) and deterministic calculation method of RBM partition function implementations.

## Install
Before you use the library, you need to compile library with

```
sh tracker/build.sh
```
which compiles c++ source codes with cython. For further usage, see tutorial.ipnb. 

You can import `tracker` module by

```
import tracker
```
on the folder resides this README.md.

## Tutorial
Tutorial is available on `tutorial.ipnb`.

## Requirements
I assume almost all functions work without caring about versions of libraries and python itself. Although the testing platform is on macOS Sierra 10.12.6 with

* python == 3.6.3
* numpy
* matplotlib
* cython == 0.27.2
* clang Apple LLVM version 9.0.0

## License
This library is distributed under MIT license. However this library uses `Eigen` as a submodule. `Eigen` itself is protected by MPL2 license[2]. If having any concern, please raise issue.


## Reference
1. Guillaume Desjardins, Yoshua Bengio, Aaron C. Couville (2011), On Tracking The Partition Function, NIPS
2. http://eigen.tuxfamily.org/index.php?title=Main_Page#License