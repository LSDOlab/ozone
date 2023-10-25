
import time
import matplotlib.pyplot as plt
# import openmdao.api as om
from ozone.api import ODEProblem, Wrap, NativeSystem
from ode_systems import ODESystemModel, ODESystemNative, ODESystemNativeSparse
import csdl
import python_csdl_backend
import numpy as np

"""
This example showcases the following:
- ability to define multiple ODE functions with coupling. For this example, the Lotka–Volterra equations have two states (x, y) which are coupled.
- ability to pass csdl parameters to your ODE function model
- multiple ways to define the ODE model itself. More info in 'ode_systems.py' where they are defined
"""

# ODE problem CLASS


class ODEProblemTest(ODEProblem):
    def setup(self):

        # Outputs. coefficients for field outputs must be defined as a CSDL variable before the integrator is created in RunModel
        self.add_field_output('field_output', state_name='x',
                              coefficients_name='coefficients')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0')
        self.add_state('x', 'dx_dt', initial_condition_name='x_0')
        self.add_times(step_vector='h')

        # Define ODE system. We have three possible choices as defined in 'ode_systems.py'. Any of the three methods yield identical results:

        self.set_ode_system(ODESystemModel)  # Uncomment for Method 1
        self.set_ode_system(ODESystemNative)  # Uncomment for Method 2
        self.set_ode_system(ODESystemNativeSparse)  # Uncomment for Method 3

# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('a')
        self.parameters.declare('num_timesteps')

    def define(self):
        num_times = self.parameters['num_timesteps']
        a = self.parameters['a']

        h_stepsize = 0.1

        # Create given inputs
        self.create_input('coefficients', np.ones(num_times+1)/(num_times+1))
        # Initial condition for state
        self.create_input('y_0', 2.0)
        self.create_input('x_0', 2.0)
        # Timestep vector
        h_vec = np.ones(num_times)*h_stepsize
        self.create_input('h', h_vec)

        # Create model containing integrator
        # We can also pass through parameters to the ODE system from this model.
        params_dict = {'a': a}

        # ODEProblem_instance
        ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times, visualization='None')
        # ODEProblem = ODEProblemTest('ExplicitMidpoint', 'time-marching', num_times, visualization='None')

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict), 'subgroup')


# Simulator Object: Note we are passing in a parameter that can be used in the ode system
sim = python_csdl_backend.Simulator(RunModel(a=2.0, num_timesteps=1000), mode='rev')
sim.prob.run_model()

# # Checktotals
print(sim.prob['field_output'])
sim.prob.check_totals(of=['field_output'], wrt=['y_0', 'x_0'], compact_print=True)
