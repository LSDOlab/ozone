
import matplotlib.pyplot as plt
import openmdao.api as om
from ozone.api import ODEProblem, NativeSystem
import csdl
import python_csdl_backend
import numpy as np

"""
This example showcases the following:
- ability to exposed intermediate variables from your ODE as outputs to the integrator.
"""


# ODE problem CLASS
class ODEProblemTest(ODEProblem):
    def setup(self):
        # Define field outputs, profile outputs, states, parameters, times

        # profile outputs are outputs from the ode integrator that are not states.
        # instead they are outputs of a function of the solved states and parameters
        self.add_profile_output('intermediate_variable_1')
        self.add_profile_output('intermediate_variable_2')

        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        self.add_parameter('a', dynamic=True, shape=(self.num_times))
        self.add_parameter('b', dynamic=True, shape=(self.num_times))
        self.add_parameter('g', dynamic=True, shape=(self.num_times))
        # If dynamic != True, it is a static parameter. i.e, the parameter used in the ODE is constant through time.
        # Therefore, the shape does not depend on the number of timesteps
        self.add_parameter('d')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0', output='solved_y')
        self.add_state('x', 'dx_dt', initial_condition_name='x_0', output='solved_x')
        self.add_times(step_vector='h')

        # To exposed variables, the profile output system is identical to the ODE model.
        # The integrator will automatically reuse the ODE model as the profile output if possible.
        self.set_ode_system(ODESystemModel, use_as_profile_output_system=True)

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
        iv1 = self.register_output('intermediate_variable_1', x*y) # intermediate variable exposed as output
        dy_dt = a*y - b*iv1
        dx_dt = g*iv1 - csdl.expand(d, n)*x


        iv2 = self.register_output('intermediate_variable_2', (dy_dt + y + x + a)**2.0) # intermediate variable exposed as output

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)


# The CSDL Model containing the ODE integrator
class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_times')

    def define(self):
        num_times = self.parameters['num_times']

        h_stepsize = 0.1

        coeffs = self.create_input('coefficients', np.ones(num_times)/(num_times))
        y_0 = self.create_input('y_0', 2.0)
        x_0 = self.create_input('x_0', 2.0)

        # Create parameter for parameters a,b,g,d
        a = np.zeros((num_times, ))
        b = np.zeros((num_times, ))
        g = np.zeros((num_times, ))
        d = 0.5  # static parameter
        for t in range(num_times):
            a[t] = 1.0 + t/num_times/5.0
            b[t] = 0.5 + t/num_times/5.0
            g[t] = 2.0 + t/num_times/5.0 

        # Add to csdl model which are fed into ODE Model
        ai = self.create_input('a', a)
        bi = self.create_input('b', b)
        gi = self.create_input('g', g)
        di = self.create_input('d', d)

        # Timestep vector
        h_vec = np.ones(num_times-1)*h_stepsize
        h = self.create_input('h', h_vec)

        # Create Model containing integrator
        ODEProblem = ODEProblemTest(
            'RK4',
            'time-marching',
            num_times,
            display='default',
            visualization='None',
        )

        self.add(ODEProblem.create_solver_model(), 'subgroup')
        
        # Intermediate variables:
        po1 = self.declare_variable('intermediate_variable_1', shape=(num_times, 1)) # intermediate variables now exposed to the outer model
        po2 = self.declare_variable('intermediate_variable_2', shape=(num_times, 1)) # intermediate variables now exposed to the outer model
        self.register_output('exposed_iv1', po1*1.0)
        self.register_output('exposed_iv2', po2*1.0)


# Simulator Object:
sim = python_csdl_backend.Simulator(RunModel(num_times=30), mode='rev')

sim.run()

# Exposed integrated variables
print(sim['exposed_iv1'])
print(sim['exposed_iv2'])

# Compute derivatives
# sim.compute_totals(of=['exposed_iv1', 'exposed_iv2', 'solved_y'], wrt=['y_0', 'x_0', 'h', 'a', 'b', 'g', 'd'])
sim.check_totals(of=['exposed_iv1', 'exposed_iv2', 'solved_y'], wrt=['y_0', 'x_0', 'h', 'a', 'b', 'g', 'd'], compact_print=True)