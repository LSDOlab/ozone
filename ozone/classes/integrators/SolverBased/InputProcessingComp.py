import csdl
import numpy as np
import scipy.sparse as sp
from ozone.classes.integrators.utils import lin_interp


class InputProcessingComp(csdl.CustomExplicitOperation):

    """
    Processes inputs to correct shapes for computation
    """

    def initialize(self):
        self.parameters.declare('parameter_dict')
        self.parameters.declare('IC_dict')
        self.parameters.declare('times')
        self.parameters.declare('state_dict')
        self.parameters.declare('stage_dict')
        self.parameters.declare('misc')
        self.parameters.declare('define_dict')
        self.parameters.declare('glm_C')

    def define(self):
        self.parameter_dict = self.parameters['parameter_dict']
        self.IC_dict = self.parameters['IC_dict']
        self.times = self.parameters['times']
        self.state_dict = self.parameters['state_dict']
        self.stage_dict = self.parameters['stage_dict']
        self.misc = self.parameters['misc']
        self.define_dict = self.parameters['define_dict']
        self.glm_C = self.parameters['glm_C']

        self.num_steps = self.misc['num_steps']
        self.num_stages = self.misc['num_stages']

        # precomputed arguments of inputs, outputs and partials
        for key in self.define_dict['inputs']:
            dd = self.define_dict['inputs'][key]
            self.add_input(**dd)
        for key in self.define_dict['outputs']:
            dd = self.define_dict['outputs'][key]
            self.add_output(**dd)
        for partials in self.define_dict['partials']:
            self.declare_derivatives(**partials)

    def compute(self, inputs, outputs):

        set_dict = {}

        # Proxy Parameters:
        for key in self.parameter_dict:
            proxy_name = self.parameter_dict[key]['proxy_name']
            if self.parameter_dict[key]['dynamic'] == False:
                outputs[proxy_name] = inputs[key]
            else:
                # *OLD dynamic parameter w/out interpolation*
                # outputs[proxy_name] = np.repeat(inputs[key], self.num_stages)

                # *NEW interpolated parameter
                temp = lin_interp(inputs[key], self.glm_C, self.num_steps, self.parameter_dict[key]['nn_shape'])
                outputs[proxy_name] = np.concatenate(tuple(temp))

            set_dict[key] = outputs[proxy_name]

        # Setting ODE parameters.
        # This allows the ODE system to not have to set inputs at every nonlinear solver iteration
        # print(self.ode_system.problem.get_val('param_b'))
        # self.ode_system.set_vars(set_dict)
        # print(self.ode_system.problem.get_val('param_b'))

        # Times vector:
        for key in self.stage_dict:
            state_name = self.stage_dict[key]['state_name']
            h_name = self.stage_dict[key]['h_name']
            outputs[h_name] = np.repeat(
                inputs[self.times['name']], self.state_dict[state_name]['num']*self.num_stages)

        for key in self.IC_dict:
            state_name = self.IC_dict[key]['state_name']
            stage_name = self.state_dict[state_name]['stage_name']
            temp = self.stage_dict[stage_name]['IC_vector_initialize']
            temp[0:self.state_dict[state_name]['num']] = inputs[key].reshape(
                (self.state_dict[state_name]['num']))

            outputs[self.IC_dict[key]['meta_name']] = temp

    def set_odesys(self, ode_sys):
        self.ode_system = ode_sys
