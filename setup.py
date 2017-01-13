try:
	from setuptools import setup, Extension
	setup
except ImportError:
	from distutils.core import setup, Extension
	setup
	
import os

os.chdir("evillens")
os.system("f2py fastell.pyf fastell.f -c")    
os.chdir("..")

setup(name='EvilLens',
	  version='1.0',
      description="Simulating gravitational lenses... Behind Yashar's back.",
	  author='Warren Morningstar',
	  author_email='wmorning@stanford.edu',
	  packages=['evillens'],
      )