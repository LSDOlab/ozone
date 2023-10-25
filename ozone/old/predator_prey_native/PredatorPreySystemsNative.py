# import openmdao.api as om
import numpy as np
from ozone.api import NativeSystem
import scipy.sparse as sp

# class ODESystem(NativeSystem):
#     # Computes f and df/dy
#     def setup(self):
#         n = self.num_nodes
#         self.add_input('y0',shape = n)
#         self.add_input('y1', shape = n)
#         # self.add_input('alpha')
#         self.add_output('dy0_dt', shape = n)
#         self.add_output('dy1_dt', shape = n)

#         rows = cols = np.arange(0,n)
#         self.declare_partial_properties('dy0_dt','y0', rows = rows, cols = cols)
#         self.declare_partial_properties('dy0_dt','y1', rows = rows, cols = cols)
#         self.declare_partial_properties('dy1_dt','y0', rows = rows, cols = cols)
#         self.declare_partial_properties('dy1_dt','y1', rows = rows, cols = cols)


#     def compute(self,inputs, outputs):
#         n = self.num_nodes
#         alpha = 1.
#         beta = 0.5
#         gamma = 2.0
#         delta = 0.5

#         outputs['dy0_dt'] = alpha*inputs['y0']- beta*np.multiply(inputs['y0'],inputs['y1'])
#         outputs['dy1_dt'] = gamma*np.multiply(inputs['y0'],inputs['y1']) - delta*inputs['y1']

#     def compute_partials(self, inputs, partials):
#         # n = self.num_nodes
#         alpha = 1.
#         beta = 0.5
#         gamma = 2.0
#         delta = 0.5
#         # print((alpha - beta*inputs['y1']))
#         partials['dy0_dt']['y0'] = (alpha - beta*inputs['y1'])
#         partials['dy0_dt']['y1'] = (- beta*inputs['y0'])
#         partials['dy1_dt']['y0'] = (gamma*inputs['y1'])
#         partials['dy1_dt']['y1'] = (gamma*inputs['y0']-delta)

class ProfileOutputSystem(NativeSystem):

    def setup(self):
        n = self.num_nodes
        self.add_input('y0', shape=n)
        self.add_output('spatial_average', shape=n)

        rows = np.arange(0, n, 1)
        cols = np.arange(0, n, 1)
        vals = np.ones(n)
        # self.partials = sp.csc_matrix((vals,(rows,cols)))

        self.declare_partial_properties('spatial_average', 'y0', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['spatial_average'] = inputs['y0']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        partials['spatial_average']['y0'] = np.ones(n)


class ODESystem(NativeSystem):
    # Computes f and df/dy
    def setup(self):
        n = self.num_nodes
        self.add_input('y0', shape=n)
        self.add_input('y1', shape=n)
        # self.add_input('alpha')
        self.add_output('dy0_dt', shape=n)
        self.add_output('dy1_dt', shape=n)

    def compute(self, inputs, outputs):
        n = self.num_nodes
        alpha = 1.
        beta = 0.5
        gamma = 2.0
        delta = 0.5
        # print(alpha*inputs['y0'])
        # print(alpha*inputs['y0'])
        outputs['dy0_dt'] = alpha*inputs['y0'] - beta*np.multiply(inputs['y0'], inputs['y1'])
        outputs['dy1_dt'] = gamma*np.multiply(inputs['y0'], inputs['y1']) - delta*inputs['y1']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        alpha = 1.
        beta = 0.5
        gamma = 2.0
        delta = 0.5
        # print((alpha - beta*inputs['y1']))
        partials['dy0_dt']['y0'] = np.diag(alpha - beta*inputs['y1'])
        partials['dy0_dt']['y1'] = np.diag(- beta*inputs['y0'])
        partials['dy1_dt']['y0'] = np.diag(gamma*inputs['y1'])
        partials['dy1_dt']['y1'] = np.diag(gamma*inputs['y0']-delta)
