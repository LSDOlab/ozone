# import openmdao.api as om
import numpy as np
from ozone.api import NativeSystem


# class ODESystem(om.ExplicitComponent):
#     # Computes f and df/dy
#     def initialize(self):
#         self.options.declare('num_nodes', types=int)

#     def setup(self):
#         n = self.options['num_nodes']

#         self.add_input('y', shape=(n, 500, 1))
#         self.add_output('dy_dt', shape=(n, 500, 1))
#         self.A = -0.01*np.tri(500, k=-1) + -0.02*np.tri(500, k=1)

#     def setup_partials(self):
#         self.declare_partials('*', '*')

#     def compute(self, inputs, outputs):
#         n = self.options['num_nodes']
#         # outputs['dy_dt'] = np.zeros((n,2,1))
#         A = self.A
#         for i in range(n):
#             outputs['dy_dt'][i] = A.dot(inputs['y'][i])

#     def compute_partials(self, inputs, partials):
#         n = self.options['num_nodes']
#         A = self.A
#         partials['dy_dt', 'y'] = np.kron(np.eye(n), A)


class ODESystemNative(NativeSystem):

    def setup(self):
        # Required every time for ODE systems or Profile Output systems
        n = self.num_nodes

        self.add_input('y', shape=(n, 500, 1))
        self.add_output('dy_dt', shape=(n, 500, 1))
        self.A = -0.01*np.tri(500, k=-1) + -0.02*np.tri(500, k=1)

    def compute(self, inputs, outputs):
        # Again, similar syntax to OpenMDAO components
        # Compute states for all (i, state) nodes where i = 0, ... n
        n = self.num_nodes
        A = self.A
        outputs['dy_dt'] = np.zeros((n, 500, 1))

        for i in range(n):
            outputs['dy_dt'][i] = A.dot(inputs['y'][i])

    def compute_partials(self, inputs, partials):
        # Again, similar syntax to OpenMDAO components
        # Compute partials for all (i, state) nodes where i = 0, ... n
        n = self.num_nodes
        A = self.A
        partials['dy_dt']['y'] = np.kron(np.eye(n), A)
        # print(partials['dy_dt']['y'].nbytes)


class ProfileSystemNative(NativeSystem):

    def setup(self):
        # Required every time for ODE systems or Profile Output systems
        n = self.num_nodes

        self.add_input('y', shape=(n, 500, 1))
        self.add_output('average', shape=(n, 1))

    def compute(self, inputs, outputs):
        # Again, similar syntax to OpenMDAO components
        # Compute states for all (i, state) nodes where i = 0, ... n
        n = self.num_nodes
        outputs['average'] = np.zeros((n, 1))

        for i in range(n):
            outputs['average'][i] = np.mean(inputs['y'][i])

    def compute_partials(self, inputs, partials):
        # Again, similar syntax to OpenMDAO components
        # Compute partials for all (i, state) nodes where i = 0, ... n
        n = self.num_nodes
        dady = 1/500*np.ones((1, 500))
        partials['average']['y'] = np.kron(np.eye(n), dady)
