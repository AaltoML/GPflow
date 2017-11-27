# GPflow

GPflow is a package for building Gaussian process models in python, using [TensorFlow](http://www.tensorflow.org). It was originally created and is now managed by [James Hensman](http://www.lancaster.ac.uk/staff/hensmanj/) and [Alexander G. de G. Matthews](http://mlg.eng.cam.ac.uk/?portfolio=alex-matthews).
The full list of [contributors](http://github.com/GPflow/GPflow/graphs/contributors) (in alphabetical order) is Artem Artemev, Rasmus Bonnevie, Alexis Boukouvalas, Ivo Couckuyt, Keisuke Fujii, Zoubin Ghahramani, David J. Harris, James Hensman, Pablo Leon-Villagra, Daniel Marthaler, Alexander G. de G. Matthews, Tom Nickson, Valentine Svensson, Mark van der Wilk. GPflow is an open source project so if you feel you have some relevant skills and are interested in contributing then please do contact us.

[![Python3.5 Status](https://travis-ci.org/GPflow/GPflow.svg?branch=master)](https://travis-ci.org/GPflow/GPflow)
[![Coverage Status](http://codecov.io/github/GPflow/GPflow/coverage.svg?branch=master)](http://codecov.io/github/GPflow/GPflow?branch=master)
[![Documentation Status](https://readthedocs.org/projects/gpflow/badge/?version=latest)](http://gpflow.readthedocs.io/en/latest/?badge=latest)

# What does GPflow do?

GPflow implements modern Gaussian process inference for composable kernels and likelihoods. The [online user manual](http://gpflow.readthedocs.io/en/latest/) contains more details. The interface follows on from [GPy](http://github.com/sheffieldml/gpy), for more discussion of the comparison see [this page](http://gpflow.readthedocs.io/en/latest/intro.html#what-s-the-difference-between-gpy-and-gpflow).

GPflow uses [TensorFlow](http://www.tensorflow.org) for runnin computations, which allows fast execution on GPUs, and is implemented in Python 3.

# Install

## 1) Quick install
GPflow can be installed by cloning the repository and running
```
pip install .
```
in the root folder. This also installs required dependencies including TensorFlow. When GPU support is needed, a manual installation of TensorFlow is recommended (next section), as one cannot rely on pip to get this running.

## 2) Alternative method
A different option to install GPflow requires installation of TensorFlow first. Please see instructions on the main TensorFlow [webpage](https://www.tensorflow.org/versions/r1.3/get_started/get_started). You will need at least version 1.3 (we aim to support the latest version). We find that for most users pip installation is the fastest way to get going. Then, for those interested in modifying the source of GPflow, we recommend
```
python setup.py develop
```
but installation should work well too:
```
python setup.py install
```
You can run the tests with `python setup.py test`.

We document the [version history](https://github.com/GPflow/GPflow/blob/master/RELEASE.md).

## Docker image

We also provide a [Docker image](https://hub.docker.com/r/gpflow/gpflow/) which can be run using

```
docker run -it -p 8888:8888 gpflow/gpflow
```

The image can be generated using our [Dockerfile](Dockerfile).

# Getting help
Please use GitHub issues to start discussion on the use of GPflow. Tagging enquiries `discussion` helps us distinguish them from bugs.

# Contributing
All constructive input is gratefully received. For more information, see the [notes for contributors](contributing.md).

# Projects using GPflow

A few projects building on GPflow and demonstrating its usage are listed below.

| Project | Description |
| --- | --- |
| [GPflowOpt](https://github.com/GPflow/GPflowOpt)       | Bayesian Optimization using GPflow. |
| [VFF](https://github.com/jameshensman/VFF)       | Variational Fourier Features for Gaussian Processes. |
| [Doubly-Stochastic-DGP](https://github.com/ICL-SML/Doubly-Stochastic-DGP)| Deep Gaussian Processes with Doubly Stochastic Variational Inference.|
| [BranchedGP](https://github.com/ManchesterBioinference/BranchedGP) | Gaussian processes with branching kernels.|

Let us know if you would like your project listed here.

# Citing GPflow

To cite GPflow, please reference the [JMLR paper](http://www.jmlr.org/papers/volume18/16-537/16-537.pdf). Sample Bibtex is given below:

```
@ARTICLE{GPflow2017,
   author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
	Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
	Ghahramani, Zoubin and Hensman, James},
    title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  month = {apr},
  volume  = {18},
  number  = {40},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v18/16-537.html}
}
```
