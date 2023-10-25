# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
import matplotlib.pyplot as plt
from benchmarksystems import ODESystem, ProfileOutputSystem
import time


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs
        self.add_times(step_vector='h')

        # ODE variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0', shape=size)

        # Output Variables
        self.add_profile_output('spatial_average', state_name='y')

        # ODE and Profile Output system
        self.ode_system = Wrap(ODESystem, options={'size': size})
        self.profile_outputs_system = Wrap(
            ProfileOutputSystem, options={'size': size})


# Script to create optimization problem
num = 3
size = 10
h_initial = 0.01
prob = om.Problem()
comp = om.IndepVarComp()
comp.add_output('coefficients', shape=num+1)
comp.add_output('y_0', shape=size)
comp.add_output('h', shape=num)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

ODEProblem_instance = ODEProblemTest(
    'RK4', 'time-marching', num_times=num, display='default', visualization=None)
comp = ODEProblem_instance.create_solver_model()
prob.model.add_subsystem('integrator', comp, promotes=['*'])
excomp = om.ExecComp('total=sum(spatial_average)',
                     spatial_average={'shape_by_conn': 'True'})
prob.model.add_subsystem('excomp', excomp, promotes=['*'])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.model.add_objective('total')
prob.model.add_design_var('y_0', lower=0.1)

prob.setup(mode='rev')
prob.set_val('coefficients', np.ones(num+1)/(num+1))
prob.set_val('y_0', 2.)
prob.set_val('h', np.ones(num)*h_initial)

start = time.time()
prob.run_driver()
end = time.time()
print('optimization run time: ', (end - start),
      'seconds \t\t', num, 'timesteps')
prob.check_totals(compact_print=True)
plt.show()
