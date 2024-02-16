import csdl
from ozone.api import NativeSystem
import numpy as np
"""
This script contains 3 possible ways on defining the same ODE function dydt = f(y) to use for the integrator
1. CSDL model

We can easily swap out these three different methods by setting
self.ode_system = 'ode system model' in the ODEProblem class
"""


# ------------------------- METHOD 1: CSDL -------------------------
# very easy to write. No need to write analytical derivatives but potentially worse performance than Native System
class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

        # We also passed in parameters to this ODE model in ODEproblem.create_solver_model() in 'run.py' which we can access here.
        self.parameters.declare('a')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.create_input('y', shape=n)
        x = self.create_input('x', shape=n)

        # Predator Prey ODE:
        a = self.parameters['a']
        b = 0.5
        g = 2.0
        d = 0.5
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y - d*x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)
