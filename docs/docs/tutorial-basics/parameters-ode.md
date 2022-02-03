---
sidebar_position: 2
---

# Parameters and Coupling

We now introduce static parameters, dynamic parameters and multiple states.

## The Problem

Let's define a new ODE,

$$
\frac{\partial{y}}{\partial{t}} = a(t)*y - b(t)*y*x, \quad y_0 = 2.0,
$$
$$
\frac{\partial{x}}{\partial{t}} = g(t)*y - d*x, \quad x_0 = 2.0.
$$

We want to integrate it from $t = 0$ to $t = 60$ using 4th order Runge-Kutta (RK4). 

Note that there are now two states ($x$ and $y$). In general, `ozone` allows any number of states with any shape.

Four parameters are introduced: $a$, $b$, $g$ and $d$. The first three, $a$, $b$, $g$, are all varying in time, hence they are **dynamic parameters**. 
$d$ does not vary in time and is a **static parameter**. In general, `ozone` allows any number of static or dynamic parameter with any shape.

With the problem defined, let's see how to implement this in `ozone`:

```py
import matplotlib.pyplot as plt
import openmdao.api as om
from ozone.api import ODEProblem
import csdl
import csdl_om
import numpy as np


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.declare_variable('y', shape=n)
        x = self.declare_variable('x', shape=n)

        # Paramters are now inputs
        a = self.declare_variable('a', shape=(n))
        b = self.declare_variable('b', shape=(n))
        g = self.declare_variable('g', shape=(n))
        d = self.declare_variable('d')

        # Predator Prey ODE:
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y - csdl.expand(d, n)*x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)


# ODE problem CLASS
class ODEProblemTest(ODEProblem):
    def setup(self):
        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        self.add_parameter('a', dynamic=True, shape=(self.num_times))
        self.add_parameter('b', dynamic=True, shape=(self.num_times))
        self.add_parameter('g', dynamic=True, shape=(self.num_times))
        # If dynamic != True, it is a static parameter. i.e, the parameter used in the ODE is constant through time.
        # Therefore, the shape does not depend on the number of timesteps
        self.add_parameter('d')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0', output='y_integrated')
        self.add_state('x', 'dx_dt', initial_condition_name='x_0', output='x_integrated')
        self.add_times(step_vector='h')

        # Define ODE
        self.set_ode_system(ODESystemModel)

# The CSDL Model containing the ODE integrator
class RunModel(csdl.Model):
    def define(self):
        num_times = 400

        h_stepsize = 0.15

        # Initial condition for state
        y_0 = self.create_input('y_0', 2.0)
        x_0 = self.create_input('x_0', 2.0)

        # Create parameter for parameters a,b,g,d
        a = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        b = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        g = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        d = 0.5  # static parameter
        for t in range(num_times):
            a[t] = 1.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
            b[t] = 0.5 + t/num_times/5.0  # dynamic parameter defined at every timestep
            g[t] = 2.0 + t/num_times/5.0  # dynamic parameter defined at every timestep

        # Add to csdl model which are fed into ODE Model
        ai = self.create_input('a', a)
        bi = self.create_input('b', b)
        gi = self.create_input('g', g)
        di = self.create_input('d', d)

        # Timestep vector
        h_vec = np.ones(num_times)*h_stepsize
        h = self.create_input('h', h_vec)

        # Create Model containing integrator
        ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times)

        self.add(ODEProblem.create_solver_model(), 'subgroup', ['*'])


# Simulator Object:
sim = csdl_om.Simulator(RunModel(), mode='rev')

sim.prob.run_model()

# Plot
plt.plot(sim.prob['y_integrated'])
plt.plot(sim.prob['x_integrated'])
plt.show()
```