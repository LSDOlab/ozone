# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
import matplotlib.pyplot as plt
from PredatorPreySystemsNative import ODESystem, ProfileOutputSystem
import time


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs
        # self.add_times(time_start = 't_start', time_end = 't_end')
        self.add_times(step_vector='h')
        # self.add_parameter('alpha')

        # ODE variables
        self.add_state('y0', 'dy0_dt', initial_condition_name='y0_0')
        self.add_state('y1', 'dy1_dt', initial_condition_name='y1_0')

        # Output Variables
        self.add_field_output(
            'field_output', coefficients_name='coefficients', state_name='y0')
        self.add_profile_output('spatial_average', state_name='y0')

        self.ode_system = ODESystem()
        self.profile_outputs_system = ProfileOutputSystem()


# Script to create optimization problem
num = 500
h_initial = 0.01
prob = om.Problem()
comp = om.IndepVarComp()
comp.add_output('coefficients', shape=num+1)
comp.add_output('y0_0')
comp.add_output('y1_0')
comp.add_output('h', shape=num)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

ODEProblem_instance = ODEProblemTest('RK4', 'time-marching', num_times=num, display='None',
                                     visualization='None')

# ODEProblem_instance = ODEProblemTest('RK4', 'time-marching checkpointing', num_times=num, display='None',
#                                      visualization='None', num_checkpoints=2)
# ODEProblem_instance = ODEProblemTest('ExplicitMidpoint', 'time-marching', num_times=num, display='None', visualization = None,error_tolerance= 0.0000001, sparse_jacobian= True)
# ODEProblem_instance = ODEProblemTest('RK4', 'solver-based', num_times=num, display='None', visualization = None)

comp = ODEProblem_instance.create_solver_model()
prob.model.add_subsystem('integrator', comp, promotes=['*'])
excomp = om.ExecComp('total=sum(spatial_average)',
                     spatial_average={'shape_by_conn': 'True'})
prob.model.add_subsystem('excomp', excomp, promotes=['*'])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.model.add_objective('total')
prob.model.add_design_var('y0_0', lower=0.1)

prob.setup(mode='rev')
prob.set_val('coefficients', np.ones(num+1)/(num+1))
prob.set_val('y0_0', 2.)
prob.set_val('y1_0', 2.)
prob.set_val('h', np.ones(num)*h_initial)

prob.run_model()
# prob.check_totals(compact_print=True)

# # print(prob.get_val('spatial_average'))
# prob.check_partials(compact_print=True)
# start = time.time()
# prob.run_model()
# end = time.time()
# print('run_model time: ', (end  - start), 'seconds \t\t', num,'timesteps')

start = time.time()
prob.run_driver()
end = time.time()
print('optimization run time: ', (end - start),
      'seconds \t\t', num, 'timesteps')
# plt.show()
