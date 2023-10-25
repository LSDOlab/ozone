# import openmdao.api as om
import numpy as np


class ODESystem(om.ExplicitComponent):
    # Computes f and df/dy
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        self.add_input('y', shape = (n,2,1))
        self.add_input('A', shape = (2,2))
        self.add_output('dy_dt',shape = (n,2,1))

    def setup_partials(self):
        self.declare_partials('*', '*')
    
    def compute(self,inputs, outputs):
        n = self.options['num_nodes']
        # outputs['dy_dt'] = np.zeros((n,2,1))
        for i in range(n):
            outputs['dy_dt'][i] = inputs['A'].dot(inputs['y'][i])

    def compute_partials(self, inputs, partials):
        n = self.options['num_nodes']
        partials['dy_dt','y'] = np.kron(np.eye(n),inputs['A'])

        # partials['dy_dt']['A'] = np.zeros((2*n,4*n))
        for i in range(n):
            # print(np.array([[inputs['y'][i][0,0], inputs['y'][i][1,0], 0.,0.],[0.,0.,inputs['y'][i][0,0], inputs['y'][i][1,0]]]))
            partials['dy_dt','A'][2*i:2*(i+1),:] = np.array([[inputs['y'][i][0,0], inputs['y'][i][1,0], 0.,0.],[0.,0.,inputs['y'][i][0,0], inputs['y'][i][1,0]]])
