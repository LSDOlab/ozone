import csdl
import python_csdl_backend
import numpy as np
from ozone.api import NativeSystem
import scipy.linalg as spln


class ODESystemNative(NativeSystem):

    def setup(self):
        n = self.num_nodes
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('y1', shape=n)
        self.add_input('param_a', shape=(1, 2))
        self.add_input('param_b', shape=n)
        self.add_output('dy_dt', shape=(n, 2, 1))
        self.add_output('dy1_dt', shape=n)

        # self.declare_partial_properties('dy1_dt', 'y1', rows = rows, cols = cols)
        self.declare_partial_properties('dy1_dt', 'param_a', empty=True)
        self.declare_partial_properties('dy1_dt', 'param_b', empty=True)
        self.declare_partial_properties('dy1_dt', 'y', empty=True)
        self.declare_partial_properties('dy_dt', 'y1', empty=True)
        self.declare_partial_properties(
            'dy1_dt', 'y1', complex_step_directional=True)

    def compute(self, inputs, outputs):
        n = self.num_nodes
        outputs['dy_dt'] = np.zeros((n, 2, 1))
        # print(n)
        # print(inputs['y'].shape)
        for i in range(n):
            outputs['dy_dt'][i] = np.array([[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [
                                           -0.1, inputs['param_b'][i][0]]]).dot(inputs['y'][i])
            # print(inputs['param_a'][0, 0], inputs['param_a'][0, 1], -0.1, inputs['param_b'][i][0])
        outputs['dy1_dt'] = -0.1*(inputs['y1']*inputs['y1'])

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        dydty = []
        dydtb = []

        partials['dy_dt']['param_a'] = np.zeros((n*2, 2))
        for i in range(n):
            partials['dy_dt']['param_a'][2*i:2*(i+1), :] = np.array(
                [[inputs['y'][i][0, 0], inputs['y'][i][1, 0]], [0., 0.]])
            dydtb.append(np.array([[0.], [inputs['y'][i][1, 0]]]))
            dydty.append(np.array(
                [[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [-0.1, inputs['param_b'][i][0]]]))

        partials['dy_dt']['y'] = spln.block_diag(*dydty)
        partials['dy_dt']['param_b'] = spln.block_diag(*dydtb)

        # COMPLEX STEP TEST:
        # partials['dy1_dt']['y1'] = -0.2*inputs['y1']


class ProfileOutputSystemNative(NativeSystem):
    # Computes f_profile that takes state as input and outputs array with specified shape
    def setup(self):
        n = self.num_nodes
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('y1', shape=(n))
        self.add_output('profile_output', shape=n)
        self.add_output('profile_output2', shape=n)

        val_s = np.ones((2*n))*0.5
        rows_s = np.concatenate((np.arange(0, n, 1), np.arange(0, n, 1)))
        cols_s = np.concatenate((np.arange(0, 2*n, 2), np.arange(1, 2*n+1, 2)))
        self.declare_partial_properties(
            'profile_output', 'y', val=val_s, rows=rows_s, cols=cols_s)

        val_s2 = np.ones(n)
        rows_s2 = np.arange(0, n)
        cols_s2 = np.arange(0, n)
        # print(val_s2)
        self.declare_partial_properties(
            'profile_output2', 'y1', val=val_s2, rows=rows_s2, cols=cols_s2)

        self.declare_partial_properties('profile_output2', 'y', empty=True)
        self.declare_partial_properties('profile_output', 'y1', empty=True)

    def compute(self, inputs, outputs):
        outputs['profile_output'] = (
            inputs['y'][:, 0, 0] + inputs['y'][:, 1, 0])/2.
        outputs['profile_output2'] = inputs['y1']
