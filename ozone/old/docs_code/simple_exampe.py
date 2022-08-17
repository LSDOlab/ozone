import csdl
import python_csdl_backend
import numpy as np
from ozone.api import ODEProblem


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Required every time for ODE systems or Profile Output systems
        n = self.parameters['num_nodes']

        # State y. All ODE states must have shape = (n, .. shape of state ...)
        y = self.create_input('y', shape=n)

        # Register output dy/dt = -y
        self.register_output('dy_dt', -y)

# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):

    def define(self):
        num_times = 30

        dt = 0.1

        # Create inputs to the ODE
        # Initial condition for state
        self.create_input('y_0', 0.5)

        # Timestep vector
        h_vec = np.ones(num_times)*dt
        self.create_input('h', h_vec)

        # Create ODEProblem class
        ode_problem = ODEProblem('RK4', 'time-marching', num_times)
        ode_problem.add_state('y', 'dy_dt', initial_condition_name='y_0', output='y_integrated')
        ode_problem.add_times(step_vector='h')
        ode_problem.set_ode_system(ODESystemModel)

        # Create CSDL Model of solver
        self.add(ode_problem.create_solver_model())


# Simulator object:
sim = python_csdl_backend.Simulator(RunModel(), mode='rev')

# Run and check derivatives
sim.prob.run_model()
print('y integrated:', sim.prob['y_integrated'])
sim.prob.check_totals(of=['y_integrated'], wrt=['y_0'], compact_print=True)
