
# import openmdao.api as om
from ozone.api import ODEProblem, Wrap, NativeSystem
import csdl
import python_csdl_backend
import numpy as np

# ODE Model with CSDL:
# Same ODE Model as coupled problem. However, the four coefficients a,b,g,d are now csdl variables that can be connected from outside


class ODESystemNative(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # Need to have ODE shapes similar as first example
        n = self.num_nodes
        self.add_input('y', shape=n)
        self.add_input('x', shape=n)
        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)

        self.add_input('a', shape=n)
        self.add_input('b', shape=n)
        self.add_input('g', shape=n)
        self.add_input('d')

        self.declare_partial_properties('dy_dt', 'g', empty=True)
        self.declare_partial_properties('dy_dt', 'd', empty=True)
        self.declare_partial_properties('dx_dt', 'a', empty=True)
        self.declare_partial_properties('dx_dt', 'b', empty=True)

    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO

    def compute(self, inputs, outputs):
        n = self.num_nodes
        a = inputs['a']
        b = inputs['b']
        g = inputs['g']
        d = inputs['d']

        # Outputs
        outputs['dy_dt'] = np.multiply(a, inputs['y']) - np.multiply(b, np.multiply(inputs['y'], inputs['x']))
        outputs['dx_dt'] = np.multiply(g, np.multiply(inputs['y'], inputs['x'])) - d*inputs['x']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        a = inputs['a']
        b = inputs['b']
        g = inputs['g']
        d = inputs['d']

        # The partials to compute.
        partials['dy_dt']['y'] = np.diag(a - b*inputs['x'])
        partials['dy_dt']['x'] = np.diag(- b*inputs['y'])
        partials['dx_dt']['y'] = np.diag(g*inputs['x'])
        partials['dx_dt']['x'] = np.diag(g*inputs['y']-d)

        partials['dy_dt']['a'] = np.diag(inputs['y'])
        partials['dy_dt']['b'] = np.diag(-np.multiply(inputs['y'], inputs['x']))
        partials['dx_dt']['d'] = np.array(-inputs['x'])
        partials['dx_dt']['g'] = np.diag(np.multiply(inputs['y'], inputs['x']))
        # The structure of partials has the following for n = self/num_nodes =  4:
        # d(dy_dt)/dy =
        # [d(dy_dt1)/dy1  0               0               0            ]
        # [0              d(dy_dt2)/dy2   0               0            ]
        # [0              0               d(dy_dt2)/dy2   0            ]
        # [0              0               0               d(dy_dt2)/dy2]
        # Hence the diagonal


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.create_input('y', shape=n)
        x = self.create_input('x', shape=n)

        # Paramters are now inputs
        a = self.create_input('a', shape=(n))
        b = self.create_input('b', shape=(n))
        g = self.create_input('g', shape=(n))
        d = self.create_input('d')

        # Predator Prey ODE:
        dy_dt = a*y - b*y*x
        # dx_dt = g*x*y - d*x
        dx_dt = g*x*y
        dx_dt = dx_dt - csdl.expand(d, n)*x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)


# ODE problem CLASS
class ODEProblemTest(ODEProblem):
    def setup(self):
        # Define field outputs, profile outputs, states, parameters, times

        # Outputs. coefficients for field outputs must be defined as an upstream variable
        self.add_field_output('field_output', state_name='x',
                              coefficients_name='coefficients')

        self.add_parameter('a', dynamic=True, shape=(num))
        self.add_parameter('b', dynamic=True, shape=(num))
        self.add_parameter('g', dynamic=True, shape=(num))
        self.add_parameter('d')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0')
        self.add_state('x', 'dx_dt', initial_condition_name='x_0')
        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.ode_system = Wrap(ODESystemModel)
        self.ode_system = ODESystemNative()

# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def define(self):

        h_stepsize = 0.05

        # Create given inputs
        # Coefficients for field output
        self.create_input('coefficients', np.ones(num+1)/(num+1))
        # Initial condition for state
        self.create_input('y_0', 2.0)
        self.create_input('x_0', 2.0)

        # Create parameter for parameters a,b,g,d
        a = np.zeros((num, 1))  # dynamic parameter defined at every timestep
        b = np.zeros((num, 1))  # dynamic parameter defined at every timestep
        g = np.zeros((num, 1))  # dynamic parameter defined at every timestep
        d = 0.5  # static parameter
        for t in range(num):
            a[t] = 1.0 + t/num/5.0  # dynamic parameter defined at every timestep
            b[t] = 0.5 + t/num/5.0  # dynamic parameter defined at every timestep
            g[t] = 2.0 + t/num/5.0  # dynamic parameter defined at every timestep

        # Add to csdl model which are fed into ODE Model
        self.create_input('a', a)
        self.create_input('b', b)
        self.create_input('g', g)
        self.create_input('d', d)

        # Timestep vector
        h_vec = np.ones(num)*h_stepsize
        self.create_input('h', h_vec)

        # Create Model containing integrator
        self.add(ODEProblem.create_solver_model(), 'subgroup')


# ODEProblem_instance
num = 100

# Integration approach: RK4 Timeamarching
ODEProblem = ODEProblemTest(
    'RK4', 'time-marching checkpointing', num_times=num, display='default', visualization='None')


# Simulator Object:
sim = python_csdl_backend.Simulator(RunModel(), mode='rev')
sim.prob.run_model()

# # Checktotals
# print(sim.prob['field_output'])
sim.prob.check_totals(of=['field_output'], wrt=[
                      'd', 'a', 'y_0', 'x_0'], compact_print=True)
