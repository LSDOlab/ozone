
import time as time
from ozone.api import ODEProblem, Wrap
import csdl
import python_csdl_backend
import numpy as np
from ode_general import ODESystemModel, ODESystemOP
from profile_general import ProfileSystemModel, ProfileSystemOP

"""
Sample ODE with robust settings:
ODE includes:
- 2x ODE's (CSDL Explicit Operations)
    - one multi-state ODE
    - one scalar ODE
- 2x Field Outputs
- 2x Profile Outputs (CSDL Explicit Operations)
- 2x Parameters:
    - static parameter
    - dynamic multi-variate parameter
"""

# ODE Problem Class. This problem class also takes in a dictionary


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Define field outputs, profile outputs, states, parameters, times

        # Outputs. coefficients for field outputs must be defined as an upstream variable
        self.add_field_output('field_output2', state_name='y1',
                              coefficients_name='coefficients')
        self.add_field_output('field_output', state_name='y',
                              coefficients_name='coefficients')
        self.add_profile_output('profile_output2', state_name='y1')
        self.add_profile_output('profile_output', state_name='y')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y1', 'dy1_dt', initial_condition_name='y1_0')
        self.add_parameter('param_b', shape=(self.num_times, 1), dynamic=True)
        self.add_parameter('param_a', shape=(1, 2))
        self.add_times(step_vector='h')
        self.add_state('y', 'dy_dt', shape=(2, 1),
                       initial_condition_name='y_0')

        # Define ODE and Profile Output systems (CSDL Explicit Operation or Models)
        self.ode_system = Wrap(ODESystemModel)
        self.profile_outputs_system = Wrap(ProfileSystemModel)

        # Define ODE and Profile Output systems (Ozone2 Native components)
        # self.ode_system = ODESystemNative()
        # self.profile_outputs_system = ProfileOutputSystemNative()


# The CSDL Model containing the ODE integrator
class ODEModel(csdl.Model):
    def define(self):

        num = 10
        h_initial = 0.2

        # Create given inputs
        self.create_input('param_a', np.array([[-0.5, -0.2]]))
        param_b = np.zeros(num)
        for i in range(num):
            param_b[i] = -i/(num*1.)
        self.create_input('coefficients', np.ones(num+1)/(num+1))
        self.create_input('param_b', param_b)
        self.create_input('y_0', np.array([[1], [1]]))
        self.create_input('y1_0', 1.)
        h_vec = np.ones(num)*h_initial
        h_vec[2] = h_vec[3]*2.0
        self.create_input('h', h_vec)

        # ODEProblem_instance
        # ODEProblem_instance = ODEProblemTest(
        #     'RK4', 'time-marching', num_times=num, display='default', visualization=None)
        ODEProblem_instance = ODEProblemTest(
            'RK4', 'time-marching', num_times=num, display='default', visualization=None)
        # ODEProblem_instance = ODEProblemTest(
        #     'RK4', 'solver-based', num_times=num, display='default', visualization='none')
        self.add(ODEProblem_instance.create_solver_model(), 'subgroup')

        # # Output
        # out = self.declare_variable('profile_output2')
        # self.register_output('out', out)


# Backend problem that can be ran.
# ODEProblem_instance = ODEProblemTest(
#     'RK4', 'time-marching', num_times=num, display='default', visualization='none')
# ODEProblem_instance.check_partials(['ProfileSystem', 'ODESystem'])
sim = python_csdl_backend.Simulator(ODEModel(), mode='rev')
tstart = time.time()
sim.prob.run_model()
print(sim.prob['field_output'])
# sim.prob.check_partials(compact_print=True)
tstart = time.time()
sim.prob.check_totals(of=['profile_output2', 'profile_output', 'field_output2', 'field_output'], wrt=[
                      'y_0', 'y1_0', 'param_b', 'param_a', 'h'], compact_print=True)
print(time.time() - tstart)
