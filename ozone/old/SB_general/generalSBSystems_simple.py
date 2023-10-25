# import openmdao.api as om
import numpy as np

class ODESystem(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('y', shape = (n,2,1))
        self.add_input('param')
        self.add_output('dy_dt', shape = (n,2,1))

    def setup_partials(self):
        self.declare_partials('dy_dt', 'y')
        self.declare_partials('dy_dt', 'param')

    def compute(self,inputs, outputs):
        n = self.options['num_nodes']
        for i in range(n):
            outputs['dy_dt'][i] = np.array([[-0.5, -0.5], [-0.1, -0.2]]).dot(inputs['y'][i])*inputs['param']


    def compute_partials(self, inputs, partials):
        n = self.options['num_nodes']
        for i in range(n):
            partials['dy_dt','y'][2*i:2*(i+1), 2*i:2*(i+1)] = np.array([[-0.5, -0.5], [-0.1, -0.2]])*inputs['param']
            partials['dy_dt','param'][2*i:2*(i+1), [0]] = np.array([[-0.5, -0.5], [-0.1, -0.2]]).dot(inputs['y'][i])