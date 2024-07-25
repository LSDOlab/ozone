# import openmdao.api as om
import numpy as np

class ODESystem(om.ExplicitComponent):
    # Computes f and df/dy
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('y0',shape = n)
        self.add_input('y1', shape = n)
        # self.add_input('alpha')
        self.add_output('dy0_dt', shape = n)
        self.add_output('dy1_dt', shape = n)

        rc = np.arange(0,n,1)
        self.declare_partials('dy0_dt','y0', rows = rc,cols = rc)
        self.declare_partials('dy0_dt','y1', rows = rc,cols = rc)
        self.declare_partials('dy1_dt','y0', rows = rc,cols = rc)
        self.declare_partials('dy1_dt','y1', rows = rc,cols = rc)
    
    def compute(self,inputs, outputs):
        n = self.options['num_nodes']
        alpha = 1.
        beta = 0.5
        gamma = 2.0
        delta = 0.5
        outputs['dy0_dt'] = alpha*inputs['y0']- beta*np.multiply(inputs['y0'],inputs['y1'])
        outputs['dy1_dt'] = gamma*np.multiply(inputs['y0'],inputs['y1']) - delta*inputs['y1']

    def compute_partials(self, inputs, partials):
        n = self.options['num_nodes']
        alpha = 1.
        beta = 0.5
        gamma = 2.0
        delta = 0.5
        partials['dy0_dt','y0'] = (alpha - beta*inputs['y1'])
        partials['dy0_dt','y1'] = (- beta*inputs['y0'])
        partials['dy1_dt','y0'] = (gamma*inputs['y1'])
        partials['dy1_dt','y1'] = (gamma*inputs['y0']-delta)

class ProfileOutputSystem(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('y0', shape = n)
        self.add_output('spatial_average', shape = n)

        rows = np.arange(0,n,1)
        cols = np.arange(0,n,1)
        vals = np.ones(n)
        self.declare_partials('spatial_average', 'y0',rows = rows, cols = cols, val = vals)

    def compute(self,inputs, outputs):
        outputs['spatial_average'] = inputs['y0']