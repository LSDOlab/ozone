import csdl
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import time


class StateComp(csdl.CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('parameter_dict')
        self.parameters.declare('IC_dict')
        self.parameters.declare('times')
        self.parameters.declare('state_dict')
        self.parameters.declare('stage_dict')
        self.parameters.declare('f2s_dict')
        self.parameters.declare('misc')
        self.parameters.declare('output_state_list')
        self.parameters.declare('define_dict')
        self.parameters.declare('stage_f_dict')

        self.parameters.declare('ODE_system')

    def define(self):
        self.parameter_dict = self.parameters['parameter_dict']
        self.IC_dict = self.parameters['IC_dict']
        self.times = self.parameters['times']
        self.state_dict = self.parameters['state_dict']
        self.stage_dict = self.parameters['stage_dict']
        self.f2s_dict = self.parameters['f2s_dict']
        self.misc = self.parameters['misc']
        self.ode_system = self.parameters['ODE_system']
        self.output_state_list = self.parameters['output_state_list']
        self.define_dict = self.parameters['define_dict']
        self.stage_f_dict = self.parameters['stage_f_dict']

        self.num_steps = self.misc['num_steps']
        self.num_stages = self.misc['num_stages']

        self.ongoingplot = self.misc['ongoingplot']
        self.visualization = self.misc['visualization']

        # Inputs: times, IC_vector, F
        # Outputs: state

        for key in self.define_dict['inputs']:
            dd = self.define_dict['inputs'][key]
            self.add_input(**dd)
        for key in self.define_dict['outputs']:
            dd = self.define_dict['outputs'][key]
            self.add_output(**dd)
        for partials in self.define_dict['partials']:
            self.declare_derivatives(**partials)

    def compute(self, inputs, outputs):
        if self.visualization == 'end':
            self.ongoingplot
            plt.clf()

        for key in self.output_state_list:
            sd = self.state_dict[key]

            ICname = self.IC_dict[sd['IC_name']]['meta_name']
            h_name = sd['h_name']
            f_name_real = sd['f_name']
            f_name = self.stage_f_dict[f_name_real]['state_name']

            t_vec = inputs[h_name]
            Fbar = inputs[f_name]
            ICvec = inputs[ICname]
            hF = np.multiply(t_vec, Fbar)
            # print((sd['ImV_inv']*(sd['B_full']*hF+ICvec)))
            outputs[sd['meta_name']] = (
                sd['ImV_inv']*(sd['B_full']*hF+ICvec)).reshape(sd['output_shape'])

            if self.visualization == 'end':
                state_flattened = outputs[sd['meta_name']].reshape(
                    (self.num_steps+1, sd['num']))
                time_vector_plot = np.arange(0, self.num_steps+1)
                time_vector_plot = np.zeros(time_vector_plot.shape)

                t_vec_state = []
                for i in range(t_vec.shape[0]):
                    if i % self.num_stages == 0:
                        t_vec_state.append(t_vec[i])
                time_vector_plot[1:] = np.cumsum(t_vec_state)
                for i in range(sd['num']):
                    plt.plot(time_vector_plot, state_flattened, label=key)
                plt.xlabel('Time')
                plt.ylabel('states')
                plt.grid(True)

        if self.visualization == 'end':
            plt.legend()
            plt.draw()
            plt.pause(0.00001)

    def compute_derivatives(self, inputs, partials):
        # start = time.time()
        for key in self.output_state_list:
            sd = self.state_dict[key]

            h_name = sd['h_name']
            f_name_real = sd['f_name']
            f_name = self.stage_f_dict[f_name_real]['state_name']

            t_vec = inputs[h_name]
            Fbar = inputs[f_name]

            # Partials for f
            partials[sd['meta_name'], f_name] = sd['ImV_invB'].dot(
                np.diagflat(t_vec))
            (i, j, val) = sp.find(sd['ImV_invB'])

            # Partials for t
            partials[sd['meta_name'], h_name] = sd['ImV_invB'].dot(
                np.diagflat(Fbar))
            # print(np.count_nonzero(partials[sd['meta_name'], h_name]), np.prod((partials[sd['meta_name'], h_name]).shape))
        # end = time.time()
        # print(end - start)
