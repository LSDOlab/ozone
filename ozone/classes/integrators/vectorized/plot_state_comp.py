import csdl
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import time


class PlotStateComp(csdl.CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('misc')
        self.parameters.declare('define_dict')

    def define(self):
        self.misc = self.parameters['misc']
        self.ongoingplot = self.misc['ongoingplot']
        self.visualization = self.misc['visualization']
        self.num_steps = self.misc['num_steps']
        self.num_stages = self.misc['num_stages']
        self.define_dict = self.parameters['define_dict']

        for key in self.define_dict:
            dd = self.define_dict[key]
            self.add_input(**dd)
        self.add_output('OZONE_PLOT_DUMMY', val=0.0)

    def compute(self, inputs, outputs):

        self.ongoingplot
        plt.clf()
        t_vec = inputs[self.define_dict['times']['name']].flatten()

        for key in self.define_dict:
            if key == 'times':
                continue
            dd = self.define_dict[key]
            shape = dd['shape']
            num = int(np.prod(shape)/(self.num_steps+1))

            if self.visualization == 'end':
                state_flattened = inputs[dd['name']].reshape(
                    (self.num_steps+1, num))
                time_vector_plot = np.arange(0, self.num_steps+1)
                time_vector_plot = np.zeros(time_vector_plot.shape)

                # t_vec_state = []
                # for i in range(t_vec.shape[0]):
                #     if i % self.num_stages == 0:
                #         t_vec_state.append(t_vec[i])
                time_vector_plot[1:] = np.cumsum(t_vec)
                for i in range(num):
                    plt.plot(time_vector_plot, state_flattened[:, i], label=key)
                plt.xlabel('Time')
                plt.ylabel('states')
                plt.grid(True)

        if self.visualization == 'end':
            plt.legend()
            plt.draw()
            plt.pause(0.00001)

        outputs['OZONE_PLOT_DUMMY'] = 0.0
