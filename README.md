[![GitHub Actions Test Badge](https://github.com/LSDOlab/ozone/actions/workflows/actions.yml/badge.svg)](https://github.com/ozone/ozone/.github)
[![Forks](https://img.shields.io/github/forks/LSDOlab/ozone.svg)](https://github.com/LSDOlab/ozone/network)
[![Issues](https://img.shields.io/github/issues/LSDOlab/ozone.svg)](https://github.com/LSDOlab/ozone/issues)

**_This repository is currently being updated from an earlier version. Continuous updates such as proper documentation and tutorials will be made over the coming weeks._**

# Introduction
`Ozone` is a Python library for solving ordinary differential equations (ODEs) within gradient-based optimization algorithms. The ODEs can be solved via one of four available solution approaches (time-marching, time-marching with checkpointing, Picard iteration, and collocation) using any explicit or implicit Runge--Kutta method. Automatic derivatives with respect to parameters and timespans are available for all approaches and methods.

`Ozone` is implemented using the [Computational System Design Language](https://github.com/LSDOlab/CSDL_alpha) (`CSDL`), a framework for building and solving optimization models. To use `ozone`, users must work within the `CSDL` framework. For guidance on how to use `CSDL`, please refer to its [documentation](https://csdl-alpha.readthedocs.io/en/latest/).

We encourage users to look at the examples:
* [Lotka Volterra](examples/simple_examples/lotka_volterra.py)
* [Van der Pol Oscillator](examples/paper_examples/2_van_der_pol_oscillator.py)
* [Lunar Ascent System](examples/paper_examples/1_lunar_ascent.py)
* [Trajectory Optimization](examples/paper_examples/3_trajectory_optimization.py)
* [Diffusion Reaction PDE](example/paper_examples/4_nonlinear_diffusion_reaction.py)

# Installation
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/LSDOlab/ozone.git
```
This should install all required packages.
If `csdl_alpha` and `modopt` failed to install:
```sh
pip install git+https://github.com/LSDOlab/csdl_alpha.git
pip install git+https://github.com/LSDOlab/modopt.git
```

# Tutorial
(Soon)

# License
This project is licensed under the terms of the **GNU Lesser General Public License v3.0**.