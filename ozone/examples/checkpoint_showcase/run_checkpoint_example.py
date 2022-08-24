import numpy as np
from ozone.api import ODEProblem, Wrap
from checkpointsystems import ODESystemNative, ProfileSystemNative
import matplotlib.pyplot as plt
import time
import python_csdl_backend
import csdl

from guppy import hpy
hp = hpy()


"""
This script runs the same ODE multiple times without checkpointing and with a different number of checkpoints to visualize memory usage.
ODE: 1 state of size 500, 300 timesteps. The df/dy Jacobian is 500x500 = 250000 elements = 2000000 Bytes to store.

run 'plot_results.py' to see a visuailzation of how the number of checkpoints can reduce memory usage.
run this script to get the data for 'plot_results.py' yourself.
"""


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs
        self.add_times(step_vector='h')

        # ODE variables
        self.add_state(
            'y', 'dy_dt', initial_condition_name='y0', shape=(500, 1))

        # Output Variables
        self.add_profile_output('average')

        # ODE and Profile Output system
        self.set_ode_system(ODESystemNative)
        self.set_profile_system(ProfileSystemNative)


class RunModel(csdl.Model):

    def initialize(self):
        self.parameters.declare('num_checkpoints')

    def define(self):
        # Reading number of checkpoints:
        nc = self.parameters['num_checkpoints']

        # Defining Inputs
        num = 300  # Number of time steps
        h_initial = 0.01  # Timesteps

        coefs = np.zeros(num+1)
        coefs[num] = 1.
        self.create_input('coefficients', val=coefs)
        self.create_input('y0', val=2*np.ones((500, 1))
                          )  # State size is 500 values
        self.create_input('h', np.ones(num-1)*h_initial)

        # Create ODE Model
        if nc == -1:
            ODEProblem_instance = ODEProblemTest(
                'ForwardEuler', 'time-marching', num, display=None, visualization=None)
        else:
            # Here is where we specify checkpointing. If the num_checkpoints argument is left empty, it automatically uses the optimal number of checkpoints
            ODEProblem_instance = ODEProblemTest(
                'ForwardEuler', 'time-marching checkpointing', num, display=None, visualization=None, num_checkpoints=nc)
        self.add(ODEProblem_instance.create_solver_model())

        average = self.declare_variable('average', shape=(300, 1))
        total = csdl.sum(average)

        self.register_output('total', total)


memory_vector_total = []
duration_vector_total = []
derivative_norm_vector_total = []
value_vector_total = []
# Run multiple times and compute average in 'plot_results.py'
for j in range(2):
    # NC Vector contains the different number of checkpoints to run ODE over
    nc_vector = np.concatenate([np.array([-1]), np.arange(0, 25, 3)])
    for i in range(5):
        nc_vector = np.append(nc_vector, 25+60*i)
    memory_vector = []
    duration_vector = []
    derivative_norm_vector = []
    value_vector = []

    #  For each different number of checkpoints, run and compute derivatives and look at memory usage.
    for nc in nc_vector:

        # Script to create optimization problem
        print('----------------------------------',
              nc, ' CHECKPOINTS ----------------------------------')

        # Simulator object
        sim = python_csdl_backend.Simulator(RunModel(num_checkpoints=nc), mode='rev')
        sim.prob.run_model()

        # Setup memory and time
        hp.setrelheap()
        t1 = time.time()

        # Take memory and time of compute totals only:
        sim.prob.run_model()
        dt_dy0 = sim.prob.compute_totals(
            of=['total'], wrt=['y0'])['total', 'y0']
        totald_current = np.linalg.norm(dt_dy0)

        # Read memory and time
        t3 = time.time()
        heap_current = hp.heap()

        total = sim.prob.get_val('total')
        time_current = t3 - t1

        print("BYTES USED:", heap_current)
        # print(heap_current.size)
        # print(dt_dy0.shape)
        print(heap_current.byid)
        memory_vector.append(heap_current.size)
        duration_vector.append(time_current)
        derivative_norm_vector.append(totald_current)
        value_vector.append(total)

    memory_vector_total.append(memory_vector)
    duration_vector_total.append(duration_vector)
    derivative_norm_vector_total.append(derivative_norm_vector)
    value_vector_total.append(value_vector)

# Save for plot_results.py
np.save('memory', memory_vector_total)
np.save('duration_vector', duration_vector_total)
np.save('derivative_norm_vector', derivative_norm_vector_total)
np.save('value_vector', value_vector_total)
np.save('nc_vector', nc_vector)

plt.show()

sim.prob.run_model()
