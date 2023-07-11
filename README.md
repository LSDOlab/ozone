Getting started
===============

ODE integration package compatible with CSDL. See [documentation site](https://lsdolab.github.io/ozone/) for more details

Introduction
------------
This package allows automatic time integration of an ODE. The user simply defines an ODE function and ``ozone`` returns a CSDL model that can be added to your problem.

Benefits
--------
``Ozone `` has the following benefits

- Automatically computes adjoint derivatives of outputs with respect to inputs through the time integration procedure.
- Modular:
    - Provides array of integration schemes:
        - Backward Euler, RK4, etc.
    - Provides multiple solver methods:
        - Time-marching, checkpointing, Picard-iteration, etc
    - Allows customizable outputs:
        - Output of integrator CSDL Model is not limited to just the full state history
    - Choice of how to define ODE function:
        - Define ODE as a CSDL Model or a function

Example
-------
Defining an ODE and adding the integrator to a CSDL Model has the following general workflow:
1. Create an ODE System defining dy_dt = f(x,y)

2. Create an ```ODEProblem``` class to define ODE settings

3. Create a CSDL Model with ```ODEProblem.create_solver_model()```

Examples can be found in ``ozone/examples``. A simple example is shown below.
```python
from ozone.api import ODEProblem
import csdl
import python_csdl_backend
import numpy as np


# STEP 1: ODE Model
# CSDL Model defining dydt = -y
class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Required every time for ODE systems or Profile Output systems
        n = self.parameters['num_nodes']

        # State y. All ODE states must have shape = (n, .. shape of state ...)
        # y = self.create_input('y', shape=n)
        y = self.declare_variable('y', shape=n)

        # What is num_nodes? n = num_nodes allows vectorization of the ODE:
        # for example, for n = 3:
        # [dy_dt1[0]]           [-y[0]]
        # [dy_dt1[1]]     =     [-y[1]]
        # [dy_dt1[2]]           [-y[2]]
        # This allows the integrator to call this model only once to evaluate the ODE function 3 times instead of calling the model 3 separate times.
        # num_nodes is purely implemented by the integrator meaning the user does not set it.

        # Compute output dy/dt = -y
        dy_dt = -y

        # Register output
        self.register_output('dy_dt', dy_dt)


# The CSDL Model containing the ODE integrator
class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_times')

    def define(self):
        num_times = self.parameters['num_times']

        dt = 0.1

        # Create inputs to the ODE
        # Initial condition for state
        self.create_input('y_0', 1.0)
        # Timestep vector
        h_vec = np.ones(num_times-1)*dt
        self.create_input('h', h_vec)

        ode_problem = ODEProblem('RK4', 'time-marching', num_times)
        ode_problem.add_state('y', 'dy_dt', initial_condition_name='y_0', output='y_integrated')
        ode_problem.add_times(step_vector='h')
        ode_problem.set_ode_system(ODESystemModel)

        # STEP 3: Create CSDL Model of intergator
        self.add(ode_problem.create_solver_model(), 'subgroup')


# Simulator object:
sim = python_csdl_backend.Simulator(RunModel(num_times=31), mode='rev')

# Run and check derivatives
sim.prob.run_model()
# sim.visualize_implementation()
print('y integrated:', sim.prob['y_integrated'])
sim.prob.check_totals(of=['y_integrated'], wrt=['y_0'], compact_print=True)
```

Installation
------------
For installation, follow these steps:

1. Install ``CSDL`` and an appropriate backend

2. Clone this repository using ``git clone`` and navigate to directory. Use ``pip install -e .`` to install.

run tests by running ``pytest`` while in root directory.


Approaches and Methods
------------------------
The first two arguments of the ODEProblem object specify what numerical integration scheme to use and how to solve it. Any combination of the two can be used.

Integration methods are the first argument of the ODEProblem object and can be one of the following:
- 'ForwardEuler'
- 'BackwardEuler'
- 'ExplicitMidpoint'
- 'ImplicitMidpoint'
- 'KuttaThirdOrder'
- 'RK4'
- 'RK6'
- 'RalstonsMethod'
- 'HeunsMethod'
- 'GaussLegendre2'
- 'GaussLegendre4'
- 'GaussLegendre6'
- 'Lobatto2'
- 'Lobatto4'
- 'RadauI3'
- 'RadauI5'
- 'RadauII3'
- 'RadauII5'
- 'Trapezoidal'
- 'AB1'
- 'AM1'
- 'BDF1'

Solver methods are the second argument of the ODEProblem object and can be one of the following:
- 'time-marching': Compute the state sequentially through timesteps.
- 'solver-based': Compute the state across timesteps in parallel
- 'time-marching checkpointing': Same as time-marching but memory usage is reduced with the added cost of slower computation time
- 'collocation': Solves for the state through an optimization problem
