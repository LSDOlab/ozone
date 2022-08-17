import matplotlib.pyplot as plt
from ozone.api import ODEProblem, Wrap, NativeSystem
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
        y = self.create_input('y', shape=n)

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

# STEP 2: ODEProblem class


class ODEProblemTest(ODEProblem):
    def setup(self):
        # User needs to define setup method
        # Define ODE from Step 2.
        # self.ode_system = Wrap(ODESystemModel)
        self.ode_system = ODEModelNS()

        # State names and timevector correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0')
        self.add_times(step_vector='h')

        # Define a field output which is a linear combination of states across timesteps weighted by the 'coefficients'
        self.add_field_output('field_output', state_name='y', coefficients_name='coefficients')

# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_times')

    def define(self):
        num_times = self.parameters['num_times']

        dt = 0.01

        # Create inputs to the ODE
        # Coefficients for field output
        self.create_input('coefficients', np.ones(num_times+1)/(num_times+1))
        # Initial condition for state
        self.create_input('y_0', 1.0)
        # Timestep vector
        h_vec = np.ones(num_times)*dt
        self.create_input('h', h_vec)

        # ODEProblem instance from Step 2:
        ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times)

        # STEP 3: Create CSDL Model of intergator
        self.add(ODEProblem.create_solver_model(), 'subgroup')


class ODEModelNS(NativeSystem):
    def setup(self):

        self.add_input('y', shape=self.num_nodes)

        self.add_output('dy_dt', shape=self.num_nodes)

    def compute(self, inputs, outputs):
        outputs['dy_dt'] = -inputs['y']*inputs['y']/2.0

    def compute_partials(self, inputs, partials):
        partials['dy_dt']['y'] = -np.diag(inputs['y'])


# Simulator object:
sim = python_csdl_backend.Simulator(RunModel(num_times=3000), mode='rev')

# Run and check derivatives
sim.prob.run_model()

x = sim.prob['field_output'].copy()
dx = sim.prob.compute_totals(of=['field_output'], wrt=['y_0'])

# sim.prob.check_totals(of=['field_output'], wrt=[
#                       'h', 'y_0'], compact_print=True)

print('linear combination of time history:', x)

for key in dx:
    print('derivative ', key, '=', dx[key])

# derivative('field_output', 'y_0') = [[0.06264671]]
