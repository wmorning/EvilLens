EvilLens
========

Simulating images and visibilities of gravitationally lensed galaxies, behind Yashar Hezaveh's back.

## Installation

For the most part, installation should be straightforward.  Just clone the repository and most things should run.  
Unfortunately, we've added a better lens model class (PowerKappa), which is non-trivial to install because it 
calls the fastell.f code from Barkana (1998).  However, this should (in principle) be able to be compiled using
f2py as follows:

f2py -c fastell.pyf fastell.f

Note that this will fail if the fortran compiler doesn't like your python distribution.  I had to re-install gcc on one 
of my laptops to get it  to work.  You may or may not have the same problem.

Either way, the above command produces the file _fastell.so which __init__.py will try to import.  So you should definitely 
play around with compiling fastell.f in a python readable format.

## Tests, demos etc

* [Deflection angle testing](http://nbviewer.ipython.org/github/wmorning/EvilLens/blob/master/examples/DeflectionTest.ipynb)
* [Plot testing](http://nbviewer.ipython.org/github/wmorning/EvilLens/blob/master/examples/PlottingTest.ipynb)
* [Image Testing](http://nbviewer.ipython.org/github/wmorning/EvilLens/blob/master/examples/LensedImageTest.ipynb)
* [Error Metric Demonstration](http://nbviewer.ipython.org/github/wmorning/EvilLens/blob/master/examples/ErrorMetricDemo.ipynb)
* [Antenna Effects Demonstration](http://nbviewer.ipython.org/github/wmorning/EvilLens/blob/master/examples/AntennaEffects.ipynb)

## Authors

* Warren Morningstar (KIPAC)
* Yashar Hezaveh (KIPAC)
* Phil Marshall (KIPAC)

## License, Credit etc

This code is being developed as part of a thesis research project, and is work in progress. If you use any of the code, please cite (Morningstar et al, in prep). The code is licensed under GPL V2.0.
