# Remove previous build
python setup.py clean
rm *.so
rm -rf __pycache__

# Construct new build
python setup.py build_ext --inplace
