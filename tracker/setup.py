from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

module = Extension("*",
	[ "wrap/tracker_wrap.pyx",
		"src/tracker.cc", "src/tracker_helper.cc", "src/math_helper.cc"],
	include_dirs = ["src/"],
	language = "c++"
)

setup(ext_modules=cythonize(module))