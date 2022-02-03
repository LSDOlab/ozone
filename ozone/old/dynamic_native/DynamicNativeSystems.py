
from ozone.api import NativeSystem
import numpy as np


class ODESystem(NativeSystem):
    def setup(self):
        n = self.num_nodes
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('param_a', shape=(1, 2))
        self.add_input('param_b', shape=n)
        self.add_output('dy_dt', shape=(n, 2, 1))

    def compute(self, inputs, outputs):
        n = self.num_nodes
        outputs['dy_dt'] = np.zeros((n, 2, 1))
        for i in range(n):
            outputs['dy_dt'][i] = np.array([[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [-0.1, inputs['param_b'][i]]]).dot(inputs['y'][i])

    def compute_partials(self, inputs, partials):
        n = self.num_nodes

        partials['dy_dt']['y'] = np.zeros((2*n, 2*n))
        partials['dy_dt']['param_b'] = np.zeros((2*n, n))
        partials['dy_dt']['param_a'] = np.zeros((2*n, 2))
        for i in range(n):
            partials['dy_dt']['y'][2*i:2*(i+1), 2*i:2*(i+1)] = np.array([[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [-0.1, inputs['param_b'][i]]])
            partials['dy_dt']['param_b'][2*i:2*(i+1), i:(i+1)] = np.array([[0.], [inputs['y'][i][1, 0]]])
            partials['dy_dt']['param_a'][2*i:2*(i+1), :] = np.array([[inputs['y'][i][0, 0], inputs['y'][i][1, 0]], [0., 0.]])


class ProfileOutputSystem(NativeSystem):
    # Computes f_profile that takes state as input and outputs array with specified shape

    def setup(self):
        n = self.num_nodes
        self.add_input('y', shape=(n, 2, 1))
        self.add_output('spatial_average', shape=n)
        self.add_output('state2', shape=n)

    def compute(self, inputs, outputs):
        n = self.num_nodes
        # print(inputs['y'].shape)
        outputs['spatial_average'] = (inputs['y'][:, 0, 0] + inputs['y'][:, 1, 0])/2.

    def compute_partials(self, inputs, partials):
        partials['spatial_average']['y'] = np.array([[0.5, 0.5]])
