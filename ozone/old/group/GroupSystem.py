# import openmdao.api as om
import numpy as np

class ODESystem_a(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes')

    # Computes f and df/dy
    def setup(self):
        n = self.options['num_nodes']
        self.add_input('y', shape = (n,2,1))
        self.add_input('A', shape = (2,2))
        self.add_output('dy_dt1',shape = (n,2,1))

    def setup_partials(self):
        self.declare_partials('*', '*')
    
    def compute(self,inputs, outputs):
        n = self.options['num_nodes']
        for i in range(n):
            outputs['dy_dt1'][i] = inputs['A'].dot(inputs['y'][i])

    def compute_partials(self, inputs, partials):
        n = self.options['num_nodes']
        for i in range(n):
            partials['dy_dt1','y'][2*i:2*(i+1),2*i:2*(i+1)] = inputs['A']
            partials['dy_dt1','A'][2*i:2*(i+1),:] = np.array([[inputs['y'][i][0,0], inputs['y'][i][1,0], 0.,0.],[0.,0.,inputs['y'][i][0,0], inputs['y'][i][1,0]]])

class ODESystem_b(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        n = self.options['num_nodes']
        self.add_input('dy_dt1', shape = (n, 2,1))
        self.add_output('dy_dt',shape = (n, 2,1))

    def setup_partials(self):
        self.declare_partials('*', '*')
    
    def compute(self,inputs, outputs):
        n = self.options['num_nodes']
        outputs['dy_dt'] = 0.5*inputs['dy_dt1']

    def compute_partials(self, inputs, partials):
        n = self.options['num_nodes']
        partials['dy_dt','dy_dt1'] = np.eye(2*n)*0.5



class ODEGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        n = self.options['num_nodes']
        self.add_subsystem('comp1', ODESystem_a(num_nodes = n), promotes = ['*'])
        self.add_subsystem('comp2', ODESystem_b(num_nodes = n), promotes = ['*'])