# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
from checkpointsystems import ODESystemNative, ProfileSystemNative
import matplotlib.pyplot as plt
import time

from guppy import hpy
hp = hpy()


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs
        self.add_times(step_vector='h')

        # ODE variables
        self.add_state(
            'y', 'dy_dt', initial_condition_name='y0', shape=(500, 1))

        # Output Variables
        # self.add_field_output(
        #     'field_output', coefficients_name='coefficients', state_name='y')
        self.add_profile_output('average', state_name='y')

        # ODE and Profile Output system
        self.ode_system = ODESystemNative()
        self.profile_outputs_system = ProfileSystemNative()


memory_vector_total = []
duration_vector_total = []
derivative_norm_vector_total = []
value_vector_total = []
for j in range(5):
    nc_vector = np.arange(0, 25, 1)
    for i in range(10):
        nc_vector = np.append(nc_vector, 25+30*i)
    memory_vector = []
    duration_vector = []
    derivative_norm_vector = []
    value_vector = []

    for nc in nc_vector:

        # Script to create optimization problem
        print('----------------------------------',
              nc, '----------------------------------')
        num = 300
        h_initial = 0.01
        prob = om.Problem()
        comp = om.IndepVarComp()
        # These outputs must match inputs defined in ODEProblemSample
        comp.add_output('coefficients', shape=num+1)
        comp.add_output('y0', shape=(500, 1))
        comp.add_output('h', shape=num)
        prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

        if nc == -1:
            ODEProblem_instance = ODEProblemTest(
                'ForwardEuler', 'time-marching', num_times=num, display=None, visualization='end')
        else:
            ODEProblem_instance = ODEProblemTest(
                'ForwardEuler', 'time-marching checkpointing', num_times=num, display=None, visualization=None, num_checkpoints=nc)
        # ODEProblem_instance = ODEProblemTest('RK4', 'solver-based',error_tolerance= 0.0000000001,num_times=num,display='default', visualization= 'end')

        comp = ODEProblem_instance.create_solver_model()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        excomp = om.ExecComp('total=abs(sum(average))',
                             average={'shape_by_conn': 'True'})
        prob.model.add_subsystem('excomp', excomp, promotes=['*'])

        prob.setup(mode='rev')
        coefs = np.zeros(num+1)
        coefs[num] = 1.
        prob.set_val('coefficients', coefs)

        y0 = 2*np.ones((500, 1))

        prob.set_val('y0', y0)
        prob.set_val('h', np.ones(num)*h_initial)
        prob.run_model()

        # Setup memory and time
        hp.setrelheap()
        t1 = time.time()

        # Take memory and time of compute totals only:
        prob.run_model()
        dt_dy0 = prob.compute_totals(of=['total'], wrt=['y0'])['total', 'y0']
        totald_current = np.linalg.norm(dt_dy0)

        # Read memory and time
        t3 = time.time()
        heap_current = hp.heap()

        print(dt_dy0.shape)
        total = prob.get_val('total')
        time_current = t3 - t1

        print(heap_current)
        print(heap_current.size)
        print(heap_current.byid)
        memory_vector.append(heap_current.size)
        duration_vector.append(time_current)
        derivative_norm_vector.append(totald_current)
        value_vector.append(total)

    memory_vector_total.append(memory_vector)
    duration_vector_total.append(duration_vector)
    derivative_norm_vector_total.append(derivative_norm_vector)
    value_vector_total.append(value_vector)

np.save('memory', memory_vector_total)
np.save('duration_vector', duration_vector_total)
np.save('derivative_norm_vector', derivative_norm_vector_total)
np.save('value_vector', value_vector_total)
np.save('nc_vector', nc_vector)

plt.show()
