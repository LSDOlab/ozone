import matplotlib.pyplot as plt
from ozone.api import ODEProblem
import csdl
import csdl_om
import python_csdl_backend
import numpy as np
import modopt
from modopt.snopt_library import SNOPT
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import time
# from pendulum_dashboard import SampleDash

# create a csdl model that tries to minimize the amount of time it takes to make a hanging pendulum go to rest
# by controlling the input torque. Equations are taken from example three in https://math.berkeley.edu/~evans/control.course.pdf


# The parent model containing the optimization problem
class IntegratorModel(csdl.Model):

    def define(self):
        num_timepoints = 50  # number of time points for integration (includes initial condition)

        # set initial conditions
        self.create_input('initial_theta', val=2.0)
        self.create_input('initial_thetadot', val=1.0)

        # set torque as an control input to the pendulum ode for every timepoint
        self.create_input('torque', val=np.ones(num_timepoints)*0.5)

        # ---VARIABLE TIME---:
        # to specify the ode integration timespan, we give a vector of timesteps.
        # for example, if we want to integrate from 0 ~ 1 seconds with 5 timesteps, we will give
        # np.array([0.2, 0.2, 0.2, 0.2, 0.2]) as the timestep vector.
        # If we set this variable as a design variable, it will automatically change them appropriately.
        val = np.ones(num_timepoints-1)*0.02
        timestep_vector = self.create_input('timestep_vector', val)
        # the final time to minimize is the sum of the timesteps
        self.register_output('final_time', csdl.sum(timestep_vector))

        # adding the ODE solver to this model
        ode = ODEProblem('RK4', 'time-marching', num_timepoints)
        ode.add_state('theta', 'dtheta_dt', output='solved_theta', initial_condition_name='initial_theta')
        ode.add_state('theta_dot', 'dthetadot_dt', output='solved_thetadot', initial_condition_name='initial_thetadot')
        ode.add_parameter('torque', dynamic=True, shape=(num_timepoints, ))
        ode.add_times(step_vector='timestep_vector')  # here we give the timestep vector
        ode.set_ode_system(ODEFunction)  # this is the model containing the ode
        self.add(ode.create_solver_model())

        # Processing solved ode states:

        # ---PATH CONSTRAINTS---:
        # the constraint is that the pendulum is at rest by the end.
        # Here is how we extract the solved ode states from the integrator:
        solved_theta = self.declare_variable('solved_theta', shape=(num_timepoints,))
        solved_thetadot = self.declare_variable('solved_thetadot', shape=(num_timepoints,))
        # here we get the final position of the ode states and set it as constraints at the end.
        final_theta = 0
        final_theta_dot = 0
        self.register_output('constraint_final_theta', solved_theta[-1] - final_theta)
        self.register_output('constraint_final_thetadot', solved_thetadot[-1] - final_theta_dot)

        # Optimization variables
        self.add_design_variable('timestep_vector', lower=0.0001, upper=0.5)  # make sure the timesteps cannot be set as negative
        self.add_design_variable('torque', upper=100, lower=-100)  # the design variable is the torque at every timestep
        self.add_constraint('constraint_final_theta', equals=0.0)  # the constraint is that the pendulum is at rest by the end
        self.add_constraint('constraint_final_thetadot', equals=0.0)  # the constraint is that the pendulum is at rest by the end
        self.add_objective('final_time')  # the objective is to minimize the time it takes to get the pendulum to rest


# The ODE model of the hanging pendulum:
# equations are taken from example three in https://math.berkeley.edu/~evans/control.course.pdf
class ODEFunction(csdl.Model):

    def initialize(self):
        self.parameters.declare('num_nodes')

    def define(self):
        num_nodes = self.parameters['num_nodes']

        # states: theta and theta dot
        theta = self.create_input('theta', shape=(num_nodes, ))
        thetadot = self.create_input('theta_dot', shape=(num_nodes, ))

        # input to ode: torque as a function of time
        torque = self.create_input('torque', shape=(num_nodes, ))

        # states time derivative
        lam = 2
        omega = 3
        self.register_output('dtheta_dt', thetadot*1.0)
        self.register_output('dthetadot_dt', -lam*thetadot - omega**2*theta + torque)


# run the optimization problem
sim = python_csdl_backend.Simulator(IntegratorModel(), mode='rev')
# dash = SampleDash()
# sim.add_recorder(dash.get_recorder())

# sim.run()
# sim.check_totals()
# exit()

# Initial run and plot
sim.run()
time_point_vector = np.cumsum(sim['timestep_vector'])
time_point_vector = np.insert(time_point_vector, 0, 0.0)
plt.plot(time_point_vector, sim['solved_theta'])
plt.plot(time_point_vector, sim['solved_thetadot'])
plt.xlabel('time')
plt.ylabel('states')
plt.title('Integrated ODE states before optimization')
plt.legend(['theta', 'theta dot'])
plt.grid()
plt.show()

# MODOPT Optimization
prob = CSDLProblem(
    problem_name='pendulum',
    simulator=sim,)

# optimizer = SLSQP(prob)
optimizer = SNOPT(prob, Major_iterations = 500)
optimization_start = time.time()
optimizer.solve()
optimization_end = time.time()
print('OPT TIME:', optimization_end - optimization_start)
# exit()
optimizer.print_results()


# Final run and plot
sim.run()
time_point_vector = np.cumsum(sim['timestep_vector'])
time_point_vector = np.insert(time_point_vector, 0, 0.0)
plt.plot(time_point_vector, sim['solved_theta'])
plt.plot(time_point_vector, sim['solved_thetadot'])
plt.xlabel('time')
plt.ylabel('states')
plt.title('Integrated ODE states after optimization')
plt.legend(['theta', 'theta dot'])
plt.grid()
plt.show()
