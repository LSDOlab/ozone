import numpy as np
from scipy import sparse as sp
from ozone.api import NativeSystem
import csdl
from scipy.linalg import block_diag


STATE_SIZE = 500
class ODESystemNative(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # Need to have ODE shapes similar as first example
        n = self.num_nodes
        self.add_input('y', shape=(n, STATE_SIZE))
        self.add_output('dy_dt', shape=(n, STATE_SIZE))

    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO

    def compute(self, inputs, outputs):
        n = self.num_nodes

        # Outputs
        outputs['dy_dt'] = -(inputs['y']*inputs['static_param'])**2

    def compute_partials(self, inputs, partials):
        n = self.num_nodes

        # The partials to compute.
        partials['dy_dt']['y'] = -np.eye(n*STATE_SIZE)


class ODESystemCSDL(csdl.Model):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):

        n = self.parameters['num_nodes']
        y = self.create_input('y', shape=(n, STATE_SIZE,))

        param = self.create_input('static_param', shape=(STATE_SIZE,))
        expanded_param = csdl.expand(param, (n, STATE_SIZE,), 'i->ai')


        dparam = self.create_input('dynamic_param', shape=(n,1))
        expanded_dparam = csdl.expand(csdl.reshape(dparam, new_shape=(n,)), (n, STATE_SIZE,), 'i->ia')

        self.register_output('dy_dt', -(y*(expanded_param+expanded_dparam))**2)
