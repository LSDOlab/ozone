from ozone.classes.integrators.vectorized_integrator import VectorBased
from ozone.classes.integrators.vectorized.MainGroup import VectorBasedGroup
import numpy as np


class Collocation(VectorBased):

    def post_setup_init(self):
        # Perform setup routines needed only for collocation.

        # For collocation, we need to all states to be outputs even if the user only needs a subset of them.
        for state_name in self.state_dict:
            if state_name not in self.output_state_list:
                self.output_state_list.append(state_name)

        super().post_setup_init()

        # create variable information for state design variables
        for stage_name in self.stage_dict:
            state_name = self.stage_dict[stage_name]['state_name']
            self.stage_dict[stage_name]['state_dv_name'] = f'state__{state_name}_dv'

        # create variable information for state design variables
        for stage_name in self.stage_dict:
            state_name = self.stage_dict[stage_name]['state_name']

            dv_size = self.state_dict[state_name]['num']*(self.num_steps)
            shape = (dv_size,)
            guess = np.linspace(
                self.state_dict[state_name]['guess'][0],
                self.state_dict[state_name]['guess'][1],
                num=self.num_steps).reshape(shape)

            state_dv_name = self.stage_dict[stage_name]['state_dv_name']
            self.state_dict[state_name]['state_dv_info'] = {
                'state_dv_name': state_dv_name,
                'shape': shape,
                'guess': guess,
            }

        self.var_order_name = {}
        comp_list = ['InputProcessComp', 'ODEComp','FieldComp', 'ProfileComp']
        for comp_name in comp_list:
            # We have a dictionary for each component that we feed through
            var_dict = self.order_variables(comp_name)
            self.var_order_name[comp_name] = var_dict

        return self.num_stage_time, self.num_steps+1

    def get_solver_model(self):

        # return collocation main model
        component = VectorBasedGroup(solution_type='collocation')
        component.add_ODEProb(self)

        return component
