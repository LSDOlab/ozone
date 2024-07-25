# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
import matplotlib.pyplot as plt
from generalSBSystems import ODESystem, ProfileOutputSystem
import time


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs

        # ODE variables
        self.add_state('y1', 'dy1_dt', initial_condition_name='y1_0')
        self.add_parameter('param_b', shape=(self.num_times, 1), dynamic=True)
        self.add_parameter('param_a', shape=(1, 2))
        self.add_times(step_vector='h')
        self.add_state('y', 'dy_dt', shape=(2, 1),
                       initial_condition_name='y_0')
        self.add_field_output('field_output', state_name='y',
                              coefficients_name='coefficients')
        self.add_field_output('field_output2', state_name='y1',
                              coefficients_name='coefficients')
        self.add_profile_output('profile_output', state_name='y')
        self.add_profile_output('profile_output2', state_name='y1')

        # ODE and Profile Output system
        self.ode_system = Wrap(ODESystem)
        self.profile_outputs_system = Wrap(ProfileOutputSystem)


# Script to create optimization problem
num = 30
h_initial = 0.005
prob = om.Problem()
comp = om.IndepVarComp()
# These outputs must match inputs defined in ODEProblemSample
comp.add_output('coefficients', shape=num+1)
comp.add_output('param_a', shape=(1, 2))
comp.add_output('param_b', shape=(num, 1))
comp.add_output('y_0', shape=(2, 1))
comp.add_output('y1_0')
comp.add_output('h', shape=num)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

# ODEProblem_instance = ODEProblemTest(
#     'Trapezoidal', 'solver-based_EC', num_times=num, display='default', visualization='end')
ODEProblem_instance = ODEProblemTest(
    'Trapezoidal', 'time-marching', num_times=num, display='default', visualization='end')
# ODEProblem_instance = ODEProblemTest('RK4', 'time-marching',num_times=num,display='default', visualization= 'end')

comp = ODEProblem_instance.create_solver_model()
prob.model.add_subsystem('comp', comp, promotes=['*'])
excomp = om.ExecComp('total=abs(sum(profile_output) - sum(profile_output2))',
                     profile_output={'shape_by_conn': 'True'},
                     profile_output2={'shape_by_conn': 'True'})
prob.model.add_subsystem('excomp', excomp, promotes=['*'])
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
prob.set_val('h', np.ones(num)*h_initial)

prob.run_model()
prob.check_totals(of=['field_output', 'field_output2', 'profile_output', 'profile_output2'], wrt=[
                  'param_a', 'param_b', 'y_0', 'y1_0', 'h'], compact_print=True)

# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'

# start = time.time()
# # prob.run_driver()
# end = time.time()
# print('optimization run time: ', (end - start),
#       'seconds \t\t', num, 'timesteps')
# plt.show()
