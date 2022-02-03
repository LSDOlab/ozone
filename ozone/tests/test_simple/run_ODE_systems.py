import numpy as np
from scipy import sparse as sp
from ozone.api import NativeSystem
import csdl
from scipy.linalg import block_diag


class ODESystemNative(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # Need to have ODE shapes similar as first example
        n = self.num_nodes
        self.add_input('y', shape=n)
        self.add_output('dy_dt', shape=n)

    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO

    def compute(self, inputs, outputs):
        n = self.num_nodes

        # Outputs
        outputs['dy_dt'] = -inputs['y']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes

        # The partials to compute.
        partials['dy_dt']['y'] = -np.eye(n)


class ODESystemCSDL(csdl.Model):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):

        n = self.parameters['num_nodes']
        y = self.create_input('y', shape=n)
        # self.register_output('dz_dt', dz_dt)
        self.register_output('dy_dt', -y)
