import csdl
import numpy as np
import scipy.sparse as sp
import time


class StageComp(csdl.CustomExplicitOperation):

    """
    Components that computes the vectorized stage
    """

    def initialize(self):
        self.parameters.declare('parameter_dict')
        self.parameters.declare('IC_dict')
        self.parameters.declare('times')
        self.parameters.declare('state_dict')
        self.parameters.declare('stage_dict')
        self.parameters.declare('f2s_dict')
        self.parameters.declare('misc')

        self.parameters.declare('ODE_system')
        self.parameters.declare('define_dict')
        self.parameters.declare('stage_f_dict')

    def define(self):
        self.parameter_dict = self.parameters['parameter_dict']
        self.IC_dict = self.parameters['IC_dict']
        self.times = self.parameters['times']
        self.state_dict = self.parameters['state_dict']
        self.stage_dict = self.parameters['stage_dict']
        self.f2s_dict = self.parameters['f2s_dict']
        self.misc = self.parameters['misc']
        self.ode_system = self.parameters['ODE_system']
        self.define_dict = self.parameters['define_dict']
        self.stage_f_dict = self.parameters['stage_f_dict']

        self.num_steps = self.misc['num_steps']
        self.num_stages = self.misc['num_stages']

        # Inputs: times, IC_vector, f
        # Outputs: stage
        # Partials: of stage wrt times, IC_vector, f

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

        for key in self.stage_dict:
            state_name = self.stage_dict[key]['state_name']
            sd = self.state_dict[state_name]

            ICname = self.IC_dict[sd['IC_name']]['meta_name']
            h_name = sd['h_name']
            f_name_real = sd['f_name']
            f_name = self.stage_f_dict[f_name_real]['res_name']

            t_vec = inputs[h_name]
            Fbar = inputs[f_name]
            ICvec = inputs[ICname]
            hF = np.multiply(t_vec, Fbar)

            stage_out_name = self.stage_dict[key]['stage_comp_out_name']
            outputs[stage_out_name] = sd['A_full']*hF + sd['UImV_inv']*(sd['B_full']*hF+ICvec)

    def compute_derivatives(self, inputs, partials):
        # dY/din:
        # start_full = time.time()
        # print('STAGE TIMINGS')
        for stage_key in self.stage_dict:
            start = time.time()
            state_name = self.stage_dict[stage_key]['state_name']
            sd = self.state_dict[state_name]

            # Times:
            f_name_real = sd['f_name']
            f_name = self.stage_f_dict[f_name_real]['res_name']

            Fbar = inputs[f_name]
            h_name = sd['h_name']
            # temp = sd['dYdH_coefficient']*sp.diags(Fbar)

            # F:
            t_vec = inputs[h_name]
            # temp = sd['dYdH_coefficient']*sp.diags(t_vec)

            # st2 = time.time()
            # print(np.diagflat(Fbar).shape)
            # print(sd['dYdH_coefficient'].shape)
            stage_out_name = self.stage_dict[stage_key]['stage_comp_out_name']

            partials[stage_out_name, h_name] = sd['dYdH_coefficient'].dot(np.diagflat(Fbar))
            partials[stage_out_name, f_name] = sd['dYdH_coefficient'].dot(np.diagflat(t_vec))
            # end = time.time()
