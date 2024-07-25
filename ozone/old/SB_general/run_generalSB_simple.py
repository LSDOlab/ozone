# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
import matplotlib.pyplot as plt
from generalSBSystems_simple import ODESystem


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs
        self.add_times(step_vector='h')

        # ODE variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0',
                       output=True, shape=(2, 1))
        self.add_parameter('param')

        # ODE and Profile Output system
        self.ode_system = Wrap(ODESystem)


# Script to create optimization problem
num = 3
h_initial = 0.05
prob = om.Problem()
comp = om.IndepVarComp()
# These outputs must match inputs defined in ODEProblemSample
comp.add_output('y_0', shape=(2, 1))
comp.add_output('param')
comp.add_output('h', shape=num)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

ODEProblem_instance = ODEProblemTest(
    'Trapezoidal', 'solver-based_EC', num_times=num, display='default', visualization=None)

comp = ODEProblem_instance.create_solver_model()

prob.model.add_subsystem('comp', comp, promotes=['*'])

prob.setup(mode='rev')
prob.set_val('y_0', np.array([[0.1], [3.]]))
prob.set_val('h', np.ones(num)*h_initial)
prob.set_val('param', 0.8)

prob.run_model()
# ODEProblem_instance.check_partials(['ODE'])
prob.check_totals(of=['y'], wrt=['y_0', 'h', 'param'], compact_print=True)
# print(prob.get_val('y'))
# prob.check_totals(of = ['y'], wrt = ['param_a'],compact_print=True)

plt.show()
