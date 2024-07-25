# import openmdao.api as om
import numpy as np
from ozone.api import ODEProblem, Wrap
import matplotlib.pyplot as plt
from om_systems import ODESystem, ProfileOutputSystem, ODESystemNative, ProfileOutputSystemNative
import time


class ODEProblemTest(ODEProblem):
    def setup(self):
        # Inputs

        # ODE variables
        self.add_field_output('field_output', state_name='y1',
                              coefficients_name='coefficients')
        self.add_state('y1', 'dy1_dt', initial_condition_name='y1_0')
        self.add_state(
            'y2', 'dy2_dt', initial_condition_name='y2_0', shape=(2,))

        # self.add_parameter('param_a',shape = (1,2))
        self.add_times(step_vector='h')
        self.add_profile_output('profile_output2', state_name='y1')
        self.add_profile_output('profile_output1', state_name='y2')

        # # ODE and Profile Output system
        self.ode_system = Wrap(ODESystem)
        self.profile_outputs_system = Wrap(ProfileOutputSystem)

        # self.ode_system = ODESystemNative()
        # self.profile_outputs_system = ProfileOutputSystemNative()


class ProblemGroup(csdl.Model):
    def setup(self):
        # Script to create optimization problem
        num = 5
        h_initial = 0.005
        # prob = om.Problem()
        # comp = om.IndepVarComp()
        # # These outputs must match inputs defined in ODEProblemSample
        # comp.add_output('coefficients', shape=num+1)

        # comp.add_output('y1_0')
        # comp.add_output('y2_0',shape=(2,))
        # comp.add_output('h', shape = num)
        # prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

        x1 = self.create_indep_var('y1_0', val=1.)
        x2 = self.create_indep_var('y2_0', val=[1., 1.])
        x3 = self.create_indep_var('h', val=np.ones(num)*h_initial)

        ODEProblem_instance = ODEProblemTest(
            'RK4', 'time-marching', num_times=num, display='default', visualization='None')
        comp = ODEProblem_instance.create_ot_group()
        comp.declare_input('y1_0')
        comp.declare_input('y2_0', shape=(2,))
        comp.declare_input('h', shape=num)

        self.add_subsystem('ode_solver', comp, promotes=['*'])
        p2 = self.declare_input('field_output')
        # obj = 1*p2
        self.register_output('obj', 1*p2)
        # prob.model.add_objective('total')
        # prob.model.add_design_var('y1_0', upper=0.)

        # prob.model = ProblemGroup()
        # prob.setup(mode = 'rev')
        # prob.set_val('coefficients', np.ones(num+1)/(num+1))
        # prob.set_val('y1_0', 1.)
        # prob.set_val('y2_0', [1.,1.])
        # prob.set_val('h',np.ones(num)*h_initial)


prob = om.Problem()
prob.model = ProblemGroup()
prob.setup(mode='rev')
prob.run_model()
# prob.check_partials(compact_print=True)
prob.check_totals(of=['profile_output2'], wrt=['h'], compact_print=True)
