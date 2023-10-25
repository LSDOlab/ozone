
import matplotlib.pyplot as plt
# import openmdao.api as om
from ozone.api import ODEProblem, Wrap, NativeSystem
import csdl
import python_csdl_backend
import numpy as np


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.create_input('y', shape=n)
        x = self.create_input('x', shape=n)

        # Predator Prey ODE:
        a = 1.1
        b = 0.4
        g = 0.1
        d = 0.4
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y - d*x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)

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

        self.ode_system = Wrap(ODESystemModel)


# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def define(self):

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
        self.add(ODEProblem.create_solver_model(), 'subgroup')

        y_0 = self.declare_variable('y_0')
        x_0 = self.declare_variable('x_0')

        # This constraint makes sure design variables add up to 30 exactly
        constraint = y_0 + x_0
        self.register_output('constraint', constraint)

        # Design variables are initial conditions (lower and upper aren't being applied for some reason)
        self.add_design_variable('y_0', lower=10.0, upper=20.0)
        self.add_design_variable('x_0', lower=10.0, upper=20.0)

        self.add_constraint('constraint', equals=30.0)

        # Objective is just field output
        self.add_objective('field_output')


# ODEProblem_instance
num = 100

# Integration approach: Timeamarching or Checkpointing
approach = 'time-marching'
ODEProblem = ODEProblemTest('RK4', approach, num_times=num, display='default', visualization='end')


# Simulator Object:
sim = python_csdl_backend.Simulator(RunModel(), mode='rev')

sim.prob.driver = om.ScipyOptimizeDriver()
sim.prob.driver.options['optimizer'] = 'SLSQP'
sim.prob.run_driver()

# Checktotals
print('CONSTRAINT:', sim.prob['constraint'])
print('OBJECTIVE:', sim.prob['field_output'])

plt.show()
