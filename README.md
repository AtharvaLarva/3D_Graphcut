# 3D Graphcut
An interactive user interface that segments cells form 3-D confocal microscope images. It utilizes a novel segmentation algorithm that utilizes graph theory to quickly and accurately segment cell nuclei.

# Setup

If you are using a windows machine, the first task is to install the latest version of [WinPython 2.7] (http://winpython.sourceforge.net/), a portable development environment for Python that contains Numpy, Scipy, and other
scientific computing related libraries. 

If you are using OSX or Linux, install [Anaconda v 2.7](https://www.continuum.io/downloads)

Next, make sure you have pip, a Python package manager in Python. Lastly, you will need to install maxflow, which is a python libarary that creates graphs and computes the maxflow-mincut algorithm for segmentation. To do this, run this command:

`pip install maxflow`

Lastly, open Spyder (The Python IDE that is pre-installed with Anaconda), open the Python file, and run it. 





