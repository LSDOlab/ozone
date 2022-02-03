import csdl
from ozone.api import NativeSystem
import numpy as np
"""
This script contains 3 possible ways on defining the same ODE function dydt = f(y) to use for the integrator
1. CSDL model
2. NativeSystem with dense partials
3. NativeSystem with sparse partials

We can easily swap out these three different methods by setting
self.ode_system = 'ode system model' in the ODEProblem class
"""


# ------------------------- METHOD 1: CSDL -------------------------
# very easy to write. No need to write analytical derivatives but potentially worse performance than Native System
class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.create_input('y', shape=n)
        x = self.create_input('x', shape=n)

        # Predator Prey ODE:
        a = 1.1
        b = 0.4
        g = 0.1
        d = 0.4
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y - d*x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)

# ------------------------- METHOD 2: NATIVESYSTEM -------------------------
# ODE Model with Native System:
# Need to define partials unlike csdl but better performance


class ODESystemNative(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # Need to have ODE shapes similar as first example
        n = self.num_nodes
        self.add_input('y', shape=n)
        self.add_input('x', shape=n)
        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)

    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO
    def compute(self, inputs, outputs):
        n = self.num_nodes
        a = 1.1
        b = 0.4
        g = 0.1
        d = 0.4

        # Outputs
        outputs['dy_dt'] = a*inputs['y'] - b * np.multiply(inputs['y'], inputs['x'])
        outputs['dx_dt'] = g * np.multiply(inputs['y'], inputs['x']) - d*inputs['x']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        a = 1.1
        b = 0.4
        g = 0.1
        d = 0.4

        # The partials to compute.
        partials['dy_dt']['y'] = np.diag(a - b*inputs['x'])
        partials['dy_dt']['x'] = np.diag(- b*inputs['y'])
        partials['dx_dt']['y'] = np.diag(g*inputs['x'])
        partials['dx_dt']['x'] = np.diag(g*inputs['y']-d)

        # The structure of partials has the following for n = self/num_nodes =  4:
        # d(dy_dt)/dy =
        # [d(dy_dt1)/dy1  0               0               0            ]
        # [0              d(dy_dt2)/dy2   0               0            ]
        # [0              0               d(dy_dt2)/dy2   0            ]
        # [0              0               0               d(dy_dt2)/dy2]
        # Hence the diagonal

# ------------------------- METHOD 3: NATIVESYSTEM -------------------------
# ODE Models with Native System allows users to customize types of partials derivatives:
# Partial derivative properties can be set in the setup method


class ODESystemNativeSparse(NativeSystem):
    def setup(self):
        n = self.num_nodes
        self.add_input('y', shape=n)
        self.add_input('x', shape=n)
        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)

        # Here we define our partial derivatives to be sparse with fixed indices for rows and columns that we define here
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partial_properties('dy_dt', 'y', rows=rows, cols=cols)
        self.declare_partial_properties('dy_dt', 'x', rows=rows, cols=cols)
        self.declare_partial_properties('dx_dt', 'y', rows=rows, cols=cols)
        self.declare_partial_properties('dx_dt', 'x', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        # The compute method is the same as METHOD 2
        n = self.num_nodes
        a = 1.1
        b = 0.4
        g = 0.1
        d = 0.4

        outputs['dy_dt'] = a*inputs['y'] - b * \
            np.multiply(inputs['y'], inputs['x'])
        outputs['dx_dt'] = g * \
            np.multiply(inputs['y'], inputs['x']) - d*inputs['x']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        a = 1.1
        b = 0.4
        g = 0.1
        d = 0.4

        # Here, we define the values of the sparse partial derivative structure defined in set up.
        partials['dy_dt']['y'] = a - b*inputs['x']
        partials['dy_dt']['x'] = - b*inputs['y']
        partials['dx_dt']['y'] = g*inputs['x']
        partials['dx_dt']['x'] = g*inputs['y']-d

        # In this case, d(dy_dt)/d(y) = spcipy.sparse.csc_matrix((a - b*inputs['x'], (rows, cols)))
