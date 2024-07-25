# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
import matplotlib.pyplot as plt
from DynamicSystems import ODESystem, ProfileOutputSystem


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs
        self.add_times(step_vector='h')
        self.add_parameter('param_a', shape=(1, 2))
        self.add_parameter('param_b', shape=(self.num_times, 1), dynamic=True)

        # ODE variables
        self.add_state('y', 'dy_dt', shape=(2, 1),
                       initial_condition_name='y_0')
        self.add_state('y1', 'dy1_dt', initial_condition_name='y1_0')

        # Output Variables
        self.add_field_output(
            'field_output', coefficients_name='coefficients', state_name='y')
        self.add_field_output(
            'field_output2', coefficients_name='coefficients', state_name='y1')
        self.add_profile_output('spatial_average', state_name='y')
        self.add_profile_output('state2', state_name='y1')

        # ODE and Profile Output system
        self.ode_system = Wrap(ODESystem)
        self.profile_outputs_system = Wrap(ProfileOutputSystem)


# Script to create optimization problem
num = 5
h_initial = 0.05
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

ODEProblem_instance = ODEProblemTest(
    'RK4', 'time-marching', num_times=num, display='default')
comp = ODEProblem_instance.create_solver_model()


prob.model.add_subsystem('comp', comp, promotes=['*'])

prob.setup(mode='rev')
prob.set_val('coefficients', np.ones(num+1)/(num+1))
prob.set_val('param_a', np.array([[-0.5, -0.2]]))
param_b = np.zeros(num)
for i in range(num):
    param_b[i] = -i/(num*1.)
prob.set_val('param_b', param_b)
prob.set_val('y_0', np.array([[1], [1]]))
prob.set_val('y1_0', 1.)
prob.set_val('h', np.ones(num)*h_initial)

prob.run_model()
ODEProblem_instance.check_partials(['ODE', 'Profile_outputs'])
prob.check_totals(of=['spatial_average', 'state2', 'field_output', 'field_output2'], wrt=[
                  'param_a', 'param_b', 'y_0', 'y1_0', 'h'], compact_print=True)
plt.show()
