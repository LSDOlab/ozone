import csdl
import numpy as np


# Profile Output Class
class ProfileSystemOP(csdl.CustomExplicitOperation):
    # Computes f_profile that takes state as input and outputs array with specified shape
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Shapes are always tuple with : (num_nodes, shape of profile output....)
        n = self.parameters['num_nodes']
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('y1', shape=(n))
        self.add_output('profile_output', shape=n)
        self.add_output('profile_output2', shape=n)

        # We can precompute partials if constant
        val_s = np.ones((2*n))*0.5
        rows_s = np.concatenate((np.arange(0, n, 1), np.arange(0, n, 1)))
        cols_s = np.concatenate((np.arange(0, 2*n, 2), np.arange(1, 2*n+1, 2)))
        self.declare_derivatives('profile_output', 'y',
                                 val=val_s, rows=rows_s, cols=cols_s)
        val_s2 = np.ones(n)
        rows_s2 = np.arange(0, n, 1)
        cols_s2 = np.arange(0, n, 1)
        self.declare_derivatives('profile_output2', 'y1',
                                 val=val_s2, rows=rows_s2, cols=cols_s2)

    def compute(self, inputs, outputs):
        # Compute partials for all (i, state) nodes where i = 0, ... n
        outputs['profile_output'] = (
            inputs['y'][:, 0, 0] + inputs['y'][:, 1, 0])/2.
        outputs['profile_output2'] = inputs['y1']


class ProfileSystemModel(csdl.Model):
    def initialize(self):
            # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Shapes are always tuple with : (num_nodes, shape of state....)
        n = self.parameters['num_nodes']
        self.create_input('y', shape=(n, 2, 1))
        self.create_input('y1', shape=n)

        self.add(ProfileSystemOP(num_nodes=n), 'Profile', promotes=['*'])

        self.declare_variable('profile_output')
        self.declare_variable('profile_output2')
