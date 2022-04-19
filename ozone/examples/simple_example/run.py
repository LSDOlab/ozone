from ozone.api import ODEProblem
import csdl
import csdl_om
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
        self.add(ode_problem.create_solver_model(), 'subgroup', ['*'])


# Simulator object:
sim = csdl_om.Simulator(RunModel(num_times=31), mode='rev')

# Run and check derivatives
sim.prob.run_model()
# sim.visualize_implementation()
print('y integrated:', sim.prob['y_integrated'])
sim.prob.check_totals(of=['y_integrated'], wrt=['y_0'], compact_print=True)
