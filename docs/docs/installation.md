---
sidebar_position: 2
---

# Installation

## Dependencies

### Python

The source code is written in Python. [Anaconda](https://www.anaconda.com/products/individual#Downloads) is recommended for installation.

### CSDL

Models are defined using the `csdl` python package. Visit the CSDL [website](https://lsdolab.github.io/csdl/) and [github repository](https://github.com/LSDOlab/csdl) for installation. Further, the [`python_csdl_backend`](https://github.com/LSDOlab/python_csdl_backend) backend is required. 

:::note Note

Ozone relies on numpy and scipy packages. Please install them from numpy and scipy. 
It is also recommended to have matplotlib installed for visualization.

:::

## Installing Ozone

To install `ozone`, first clone the repository and install using `pip`.
From the terminal/command line,

```sh
git clone https://github.com/lsdolab/ozone.git
pip install -e ozone/
```

## Testing
To run unit tests, navigate to the `ozone` directory and run

```sh
pytest
```
Note that these tests involve solving ODEs and may take a few minutes to complete all 44 tests.

Running `simple_example/run.py` without errors can also be a quick test.