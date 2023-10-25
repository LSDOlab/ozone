
import matplotlib.pyplot as plt
# import openmdao.api as om
from ozone.api import ODEProblem, Wrap, NativeSystem
from ode_systems import ODESystemModel, ODESystemNative, ODESystemNativeSparse
import csdl
import python_csdl_backend
import numpy as np


# ODE problem CLASS
class ODEProblemTest(ODEProblem):
    def setup(self):
        # Define field outputs, profile outputs, states, parameters, times

        # Outputs. coefficients for field outputs must be defined as an upstream variable
        self.add_field_output('field_output', state_name='y',
                              coefficients_name='coefficients')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0')
        self.add_state('x', 'dx_dt', initial_condition_name='x_0')
        self.add_times(step_vector='h')

        # Define ODE system. We have three possible choices as defined in ode_systems.py. Any of the three methods yield identical results:

        # self.ode_system = Wrap(ODESystemModel)      # Uncomment for Method 1
        self.ode_system = ODESystemNative()           # Uncomment for Method 2
        # self.ode_system = ODESystemNativeSparse()   # Uncomment for Method 3

# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('a')
        self.parameters.declare('p')

    def define(self):
        a = self.parameters['a']
        p = self.parameters['p']

        h_stepsize = 0.15

        # Create given inputs
        # Coefficients for field output
        # temp = np.zeros(num+1)
        # temp[-1] = 1.0
        # self.create_input('coefficients', temp)
        self.create_input('coefficients', np.ones(num+1)/(num+1))
        # Initial condition for state
        self.create_input('y_0', 20.0)
        self.create_input('x_0', 20.0)
        # Timestep vector
        h_vec = np.ones(num)*h_stepsize
        self.create_input('h', h_vec)

        # Create model containing integrator
        ODE_params = {'a': a}
        profile_params = {'p': p}

        self.add(ODEProblem.create_solver_model(ODE_parameters=ODE_params, profile_parameters=profile_params), 'subgroup')


# ODEProblem_instance
num = 100

# Integration approach: Timemarching or Checkpointing
approach = 'time-marching'
ODEProblem = ODEProblemTest('RK4', approach, num_times=num, display='default', visualization='end', implicit_solver_jvp='direct', implicit_solver_fwd='direct')
# ODEProblem = ODEProblemTest('RK4', approach, num_times=num, display='default', visualization='None')


# Simulator Object:
sim = python_csdl_backend.Simulator(RunModel(a=2.0, p=[1.0, 2.0]), mode='rev')
sim.prob.run_model()

# Checktotals
print(sim.prob['field_output'])
sim.prob.check_totals(of=['field_output'], wrt=[
                      'y_0', 'x_0'], compact_print=True)

plt.show()
