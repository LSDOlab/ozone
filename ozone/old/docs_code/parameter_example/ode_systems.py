import numpy as np
from ozone.api import NativeSystem
import csdl


class ODESystemNative(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # Need to have ODE shapes similar as first example
        n = self.num_nodes
        self.add_input('y', shape=n)
        self.add_input('x', shape=n)
        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)

        self.add_input('a', shape=n)
        self.add_input('b', shape=n)
        self.add_input('g', shape=n)
        self.add_input('d')

        self.declare_partial_properties('dy_dt', 'g', empty=True)
        self.declare_partial_properties('dy_dt', 'd', empty=True)
        self.declare_partial_properties('dx_dt', 'a', empty=True)
        self.declare_partial_properties('dx_dt', 'b', empty=True)

        # self.declare_partial_properties('*', '*', complex_step_directional=True)

    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO

    def compute(self, inputs, outputs):
        n = self.num_nodes
        a = inputs['a']
        b = inputs['b']
        g = inputs['g']
        d = inputs['d']

        # Outputs
        outputs['dy_dt'] = np.multiply(a, inputs['y']) - np.multiply(b, np.multiply(inputs['y'], inputs['x']))
        outputs['dx_dt'] = np.multiply(g, np.multiply(inputs['y'], inputs['x'])) - d*inputs['x']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        a = inputs['a']
        b = inputs['b']
        g = inputs['g']
        d = inputs['d']

        # The partials to compute.
        partials['dy_dt']['y'] = np.diag(a - b*inputs['x'])
        partials['dy_dt']['x'] = np.diag(- b*inputs['y'])
        partials['dx_dt']['y'] = np.diag(g*inputs['x'])
        partials['dx_dt']['x'] = np.diag(g*inputs['y']-d)

        partials['dy_dt']['a'] = np.diag(inputs['y'])
        partials['dy_dt']['b'] = np.diag(-np.multiply(inputs['y'], inputs['x']))
        partials['dx_dt']['d'] = np.array(-inputs['x'])
        partials['dx_dt']['g'] = np.diag(np.multiply(inputs['y'], inputs['x']))
        # The structure of partials has the following for n = self.num_nodes =  4:
        # d(dy_dt)/dy =
        # [d(dy_dt1)/dy1  0               0               0            ]
        # [0              d(dy_dt2)/dy2   0               0            ]
        # [0              0               d(dy_dt2)/dy2   0            ]
        # [0              0               0               d(dy_dt2)/dy2]
        # Hence the diagonal


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        # y = self.create_input('y', shape=n)
        # x = self.create_input('x', shape=n)

        # # Paramters are now inputs
        # a = self.create_input('a', shape=(n))
        # b = self.create_input('b', shape=(n))
        # g = self.create_input('g', shape=(n))
        # d = self.create_input('d')

        y = self.declare_variable('y', shape=n)
        x = self.declare_variable('x', shape=n)

        # Paramters are now inputs
        a = self.declare_variable('a', shape=(n))
        b = self.declare_variable('b', shape=(n))
        g = self.declare_variable('g', shape=(n))
        d = self.declare_variable('d')

        # Predator Prey ODE:
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y - csdl.expand(d, n)*x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)
