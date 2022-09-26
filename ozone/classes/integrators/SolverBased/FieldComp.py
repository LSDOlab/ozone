import csdl
import numpy as np
import time


class FieldComp(csdl.CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('parameter_dict')
        self.parameters.declare('IC_dict')
        self.parameters.declare('times')
        self.parameters.declare('state_dict')
        self.parameters.declare('field_output_dict')
        self.parameters.declare('f2s_dict')
        self.parameters.declare('misc')
        self.parameters.declare('define_dict')

    def define(self):
        self.parameter_dict = self.parameters['parameter_dict']
        self.IC_dict = self.parameters['IC_dict']
        self.times = self.parameters['times']
        self.state_dict = self.parameters['state_dict']
        self.field_output_dict = self.parameters['field_output_dict']
        self.f2s_dict = self.parameters['f2s_dict']
        self.misc = self.parameters['misc']
        self.define_dict = self.parameters['define_dict']

        self.num_steps = self.misc['num_steps']
        self.num_stages = self.misc['num_stages']

        # Inputs: State, Coefficients (No partials)
        # Outputs: Field_output_dict

        for key in self.define_dict['inputs']:
            dd = self.define_dict['inputs'][key]
            # print('ode_comp input: ', dd['name'], key)
            self.add_input(**dd)
        for key in self.define_dict['outputs']:
            dd = self.define_dict['outputs'][key]
            self.add_output(**dd)
        for partials in self.define_dict['partials']:
            self.declare_derivatives(**partials)

    def compute(self, inputs, outputs):

        for key in self.field_output_dict:
            state_name = self.field_output_dict[key]['state_name']
            sd = self.state_dict[state_name]
            outputs[key] = np.einsum('i,i...->...', inputs[self.field_output_dict[key]['coefficients_name']], inputs[sd['meta_name']])

    def compute_derivatives(self, inputs, partials):

        for key in self.field_output_dict:
            state_name = self.field_output_dict[key]['state_name']
            sd = self.state_dict[state_name]

            partials[key, sd['meta_name']] = np.repeat(
                inputs[self.field_output_dict[key]['coefficients_name']], sd['num'])
