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
        self.add_input('x', shape=n)

        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)

        self.add_input('a', shape=n)
        self.add_input('b', shape=n)
        self.add_input('g', shape=n)
        self.add_input('d')
        self.add_input('e', shape=(n, 2, 2))

        self.add_input('z', shape=(n, 2, 2))
        self.add_output('dz_dt', shape=(n, 2, 2))

        self.declare_partial_properties('dy_dt', 'g', empty=True)
        self.declare_partial_properties('dy_dt', 'd', empty=True)
        self.declare_partial_properties('dy_dt', 'z', empty=True)
        self.declare_partial_properties('dy_dt', 'e', empty=True)

        self.declare_partial_properties('dx_dt', 'a', empty=True)
        self.declare_partial_properties('dx_dt', 'b', empty=True)
        self.declare_partial_properties('dx_dt', 'z', empty=True)
        self.declare_partial_properties('dx_dt', 'e', empty=True)
        self.declare_partial_properties('dx_dt', 'x', complex_step_directional=True)

        self.declare_partial_properties('dz_dt', 'b', empty=True)
        self.declare_partial_properties('dz_dt', 'g', empty=True)
        self.declare_partial_properties('dz_dt', 'd', empty=True)
        self.declare_partial_properties('dz_dt', 'x', sparse=True)

        self.dzx = sp.csc_matrix(np.kron(np.eye(n), np.array([[0.5], [0.5], [0.5], [0.5]])))
        # self.declare_partial_properties('dz_dt', 'y', empty=True)
        # self.declare_partial_properties('dz_dt', 'a', empty=True)
        # self.declare_partial_properties('dz_dt', 'x', empty=True)

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

        outputs['dz_dt'] = np.zeros((n, 2, 2))
        # for key in inputs:
        #     print(key, inputs[key])
        for i in range(n):
            outputs['dz_dt'][i, :, :] = -inputs['z'][i, :, :]/3.0+(inputs['y'][i]**2)/2.0+(inputs['a'][i])*inputs['e'][i, :, :]+inputs['x'][i]/2.0

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
        # partials['dx_dt']['x'] = np.diag(g*inputs['y']-d)

        partials['dy_dt']['a'] = np.diag(inputs['y'])
        partials['dy_dt']['b'] = np.diag(-np.multiply(inputs['y'], inputs['x']))
        partials['dx_dt']['d'] = np.array(-inputs['x'])
        partials['dx_dt']['g'] = np.diag(np.multiply(inputs['y'], inputs['x']))

        partials['dz_dt']['z'] = -np.eye(4*n)/3.0

        list_block_y = []
        list_block_a = []
        list_block_e = []
        for i in range(n):
            list_block_y.append(np.array([[inputs['y'][i]], [inputs['y'][i]], [inputs['y'][i]], [inputs['y'][i]]]))
            list_block_a.append(inputs['e'][i, :, :].reshape(4, 1))
            list_block_e.append(np.eye(4)*inputs['a'][i])

        partials['dz_dt']['y'] = block_diag(*list_block_y)
        partials['dz_dt']['a'] = block_diag(*list_block_a)
        partials['dz_dt']['e'] = block_diag(*list_block_e)
        partials['dz_dt']['x'] = self.dzx
        # partials['dz_dt']['x'] = np.kron(np.eye(n), np.array([[0.5], [0.5], [0.5], [0.5]]))

        # print(partials['dz_dt']['y'])

        # The structure of partials has the following for n = self.num_nodes =  4:
        # d(dy_dt)/dy =
        # [d(dy_dt1)/dy1  0               0               0            ]
        # [0              d(dy_dt2)/dy2   0               0            ]
        # [0              0               d(dy_dt2)/dy2   0            ]
        # [0              0               0               d(dy_dt2)/dy2]
        # Hence the diagonal


class ODESystemCSDL(csdl.Model):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        n = self.parameters['num_nodes']

        # y = self.create_input('y', shape=n)
        # x = self.create_input('x', shape=n)
        # a = self.create_input('a', shape=n)
        # b = self.create_input('b', shape=n)
        # g = self.create_input('g', shape=n)
        # d = self.create_input('d')
        # e = self.create_input('e', shape=(n, 2, 2))
        # z = self.create_input('z', shape=(n, 2, 2))

        y = self.declare_variable('y', shape=n)
        x = self.declare_variable('x', shape=n)
        a = self.declare_variable('a', shape=n)
        b = self.declare_variable('b', shape=n)
        g = self.declare_variable('g', shape=n)
        d = self.declare_variable('d')
        e = self.declare_variable('e', shape=(n, 2, 2))
        z = self.declare_variable('z', shape=(n, 2, 2))

        dy_dt = a*y - b*y*x
        dx_dt = g*x*y-csdl.expand(d, n)*x

        dz_dt = self.create_output('dz_dt', shape=(n, 2, 2))
        for i in range(n):
            temp_y = y[i]**2
            temp_a = a[i]
            temp_x = x[i]
            dz_dt[i, :, :] = -z[i, :, :]/3.0+csdl.expand(temp_y, (1, 2, 2))/2.0 + csdl.expand(temp_a, (1, 2, 2))*e[i, :, :] + csdl.expand(temp_x, (1, 2, 2))/2.0

        # self.register_output('dz_dt', dz_dt)
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)


class POSystemNS(NativeSystem):

    def setup(self):
        # Need to have ODE shapes similar as first example
        n = self.num_nodes
        self.add_input('x', shape=n)
        self.add_input('y', shape=n)
        self.add_input('z', shape=(n, 2, 2))

        self.add_output('profile_output_x', shape=(n))
        self.add_output('profile_output_z', shape=(n))

        c_z = np.arange(3, n*4+3, 4)
        r_z = np.arange(0, n, 1)
        v_z = np.ones(n)

        rc_x = np.arange(0, n, 1)
        v_x = np.ones(n)

        self.declare_partial_properties('profile_output_z', 'z', rows=r_z, cols=c_z, vals=v_z)
        self.declare_partial_properties('profile_output_z', 'x', empty=True)
        self.declare_partial_properties('profile_output_z', 'y', empty=True)

        self.declare_partial_properties('profile_output_x', 'x', rows=rc_x, cols=rc_x, vals=v_x)
        self.declare_partial_properties('profile_output_x', 'z', empty=True)

    def compute(self, inputs, outputs):
        outputs['profile_output_z'] = inputs['z'][:, 1, 1].flatten()
        outputs['profile_output_x'] = inputs['x'] + inputs['y']*inputs['y']

    def compute_partials(self, inputs, partials):
        partials['profile_output_x']['y'] = 2*np.diag(inputs['y'])
