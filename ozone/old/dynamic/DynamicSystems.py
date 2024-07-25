# import openmdao.api as om
import numpy as np

class ODESystem(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('y', shape = (n,2,1))
        self.add_input('y1', shape = n)
        self.add_input('param_a',shape = (1,2))
        self.add_input('param_b', shape = n)
        self.add_output('dy_dt', shape = (n, 2,1))
        self.add_output('dy1_dt', shape = n)

    def setup_partials(self):
        self.declare_partials('dy_dt', 'y')
        self.declare_partials('dy_dt', 'param_a')
        self.declare_partials('dy_dt', 'param_b')
        self.declare_partials('dy1_dt', 'y1')

    def compute(self,inputs, outputs):
        n = self.options['num_nodes']
        for i in range(n):
            outputs['dy_dt'][i] = np.array([[inputs['param_a'][0,0], inputs['param_a'][0,1]], [-0.1, inputs['param_b'][i]]]).dot(inputs['y'][i])
        outputs['dy1_dt'] = 0.01*inputs['y1']**4


    def compute_partials(self, inputs, partials):
        n = self.options['num_nodes']
        for i in range(n):
            partials['dy_dt','y'][2*i:2*(i+1), 2*i:2*(i+1)] = np.array([[inputs['param_a'][0,0], inputs['param_a'][0,1]], [-0.1, inputs['param_b'][i]]])
            partials['dy_dt','param_a'][2*i:2*(i+1),:] = np.array([[inputs['y'][i][0,0], inputs['y'][i][1,0]],[0.,0.]])
            partials['dy_dt','param_b'][2*i:2*(i+1), i:(i+1)] = np.array([[0.],[inputs['y'][i][1,0]]])
        partials['dy1_dt','y1'] = np.diag(0.04*inputs['y1']**3)


class ProfileOutputSystem(om.ExplicitComponent):
    
    # Computes f_profile that takes state as input and outputs array with specified shape
    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('y',shape = (n, 2,1))
        self.add_input('y1', shape = n)
        self.add_output('spatial_average', shape = n)
        self.add_output('state2', shape = n)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self,inputs, outputs):
        n = self.options['num_nodes']
        outputs['spatial_average'] = (inputs['y'][:, 0,0] + inputs['y'][:, 1,0])/2.
        outputs['state2'] = inputs['y1']

    def compute_partials(self, inputs, partials):
        n = self.options['num_nodes']
        partials['spatial_average', 'y'] = np.array([[0.5, 0.5]])
        partials['state2','y1'] = 1.