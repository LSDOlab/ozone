# import openmdao.api as om
import numpy as np
import scipy.sparse as sp

class ODESystem(om.ExplicitComponent):
    # Computes f and df/dy
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('size', types = int)

    def setup(self):
        n = self.options['num_nodes']
        size = self.options['size']
        self.add_input('y',shape = (n, size))
        self.add_output('dy_dt', shape =  (n,size))

    def setup_partials(self):
        self.declare_partials('*', '*')
    
    def compute(self,inputs, outputs):
        alpha = -0.1
        # print(alpha*inputs['y']+alpha*inputs['y']**2)
        outputs['dy_dt'] = alpha*(inputs['y']+inputs['y']**2)

    def compute_partials(self, inputs, partials):
        n = self.options['num_nodes']
        size = self.options['size']
        alpha = -0.1
        # print((alpha*(np.ones((n,size))+2*inputs['y'])).flatten())
        partials['dy_dt','y'] = np.diag((alpha*(np.ones((n,size))+2*inputs['y'])).flatten())
        # partials['dy_dt','y'] = sp.lil_matrix(partials['dy_dt','y'])


class ProfileOutputSystem(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('size', types=int)
        self.options.declare('num_nodes', types=int)

    def setup(self):
        size = self.options['size']
        n = self.options['num_nodes']
        self.add_input('y', shape = (n, size))
        self.add_output('spatial_average', shape = n)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self,inputs, outputs):
        size = self.options['size']
        outputs['spatial_average'] = np.mean(inputs['y'])

    def compute_partials(self, inputs, partials):
        size = self.options['size']
        partials['spatial_average','y'] = np.ones(size)/size