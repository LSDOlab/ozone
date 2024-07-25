# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
import matplotlib.pyplot as plt
from general_systems import ODESystem, ProfileOutputSystem, ODESystemNative, ProfileOutputSystemNative
import time

# We need to create an ODEProblem class and write the setup method
# def setup is where to declare input and output variables. Similar to Ozone's ODEfunction class.


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Define field outputs, profile outputs, states, parameters, times
        self.add_field_output('field_output2', state_name='y1',
                              coefficients_name='coefficients')
        self.add_field_output('field_output', state_name='y',
                              coefficients_name='coefficients')
        self.add_profile_output('profile_output2', state_name='y1')
        self.add_profile_output('profile_output', state_name='y')
        self.add_state('y1', 'dy1_dt', initial_condition_name='y1_0')

        self.add_parameter('param_b', shape=(self.num_times, 1), dynamic=True)
        self.add_parameter('param_a', shape=(1, 2))
        self.add_times(step_vector='h')
        self.add_state('y', 'dy_dt', shape=(2, 1),
                       initial_condition_name='y_0')

        # Define ODE and Profile Output systems (OpenMDAO components)
        # self.ode_system = Wrap(ODESystem)
        # self.profile_outputs_system = Wrap(ProfileOutputSystem)

        # Define ODE and Profile Output systems (Ozone2 Native components)
        self.ode_system = ODESystemNative()
        self.profile_outputs_system = ProfileOutputSystemNative()


# Script to create optimization problem
num = 500
h_initial = 0.2
prob = om.Problem()

# These outputs must match inputs defined in ODE Problem Sample
comp = om.IndepVarComp()
comp.add_output('coefficients', shape=num+1)
comp.add_output('param_a', shape=(1, 2))
comp.add_output('param_b', shape=(num, 1))
comp.add_output('y_0', shape=(2, 1))
comp.add_output('y1_0')
comp.add_output('h', shape=num)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

# ODE problem creates OpenMDAO component/group
# ODEProblem_instance = ODEProblemTest('RK4', 'solver-based',num_times=num,display='default', visualization= 'end')
ODEProblem_instance = ODEProblemTest(
    'ForwardEuler', 'time-marching', num_times=num, display=None, visualization='None')
ODEProblem_instance = ODEProblemTest(
    'ForwardEuler', 'time-marching checkpointing', num_times=num, display=None, visualization='None', num_checkpoints=1)
# ODEProblem_instance = ODEProblemTest(
#     'RK4', 'time-marching', num_times=num, display='default', visualization='None')
# ODEProblem_instance = ODEProblemTest(
#     'RK4', 'solver-based', num_times=num, display='default', visualization='None')

comp = ODEProblem_instance.create_solver_model()
prob.model.add_subsystem('comp', comp, promotes=['*'])

# Final objective function is difference between two profile outputs
excomp = om.ExecComp('total=abs(sum(profile_output) - sum(profile_output2))',
                     profile_output={'shape_by_conn': 'True'},
                     profile_output2={'shape_by_conn': 'True'})
prob.model.add_subsystem('excomp', excomp, promotes=['*'])

# Optimizations setup
prob.model.add_objective('total')
prob.model.add_design_var('param_b', upper=0.)
prob.setup(mode='rev')
prob.set_val('param_a', np.array([[-0.5, -0.2]]))
param_b = np.zeros(num)
for i in range(num):
    param_b[i] = -i/(num*1.)
prob.set_val('coefficients', np.ones(num+1)/(num+1))
prob.set_val('param_b', param_b)
prob.set_val('y_0', np.array([[1], [1]]))
prob.set_val('y1_0', 1.)
h_vec = np.ones(num)*h_initial
h_vec[2] = h_vec[3]*3.0
prob.set_val('h', h_vec)

# Run and check totals

start = time.time()
prob.run_model()
print(prob.compute_totals())
end = time.time()
print(end - start)
