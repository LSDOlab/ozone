# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
from Simplelinearsystem import ODESystem
import matplotlib.pyplot as plt
import time


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs
        self.add_times(step_vector='h')
        self.add_parameter('A', shape=(2, 2))

        # ODE variables
        self.add_state('y', 'dy_dt', initial_condition_name='y0', shape=(2, 1))

        # Output Variables
        self.add_field_output(
            'field_output', coefficients_name='coefficients', state_name='y')

        # ODE and Profile Output system
        self.ode_system = Wrap(ODESystem)


# Script to create optimization problem
num = 30
h_initial = 0.01
prob = om.Problem()
comp = om.IndepVarComp()
# These outputs must match inputs defined in ODEProblemSample
comp.add_output('coefficients', shape=num+1)
comp.add_output('A', shape=(2, 2))
comp.add_output('y0', shape=(2, 1))
comp.add_output('h', shape=num)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

ODEProblem_instance = ODEProblemTest(
    'RK4', 'time-marching', num_times=num, display='default', visualization=None)
# ODEProblem_instance = ODEProblemTest('RK4', 'solver-based',error_tolerance= 0.0000000001,num_times=num,display='default', visualization= 'end')

comp = ODEProblem_instance.create_solver_model()
prob.model.add_subsystem('comp', comp, promotes=['*'])

prob.setup(mode='rev')
coefs = np.zeros(num+1)
coefs[num] = 1.
prob.set_val('coefficients', coefs)

prob.set_val('A', np.array([[-1, -0.5], [-2, -2]]))
prob.set_val('y0', np.array([[2.], [2.]]))
prob.set_val('h', np.ones(num)*h_initial)

t1 = time.time()
prob.run_model()
prob.check_totals(of=['field_output'], wrt=['y0', 'h', 'A'])
print('run time: ', time.time() - t1)

print(prob.get_val('field_output'))

plt.show()
