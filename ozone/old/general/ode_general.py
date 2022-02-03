import csdl
import numpy as np

#  ODE System Operation Class


class ODESystemOP(csdl.CustomExplicitOperation):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        print('DEFINE')
        # Shapes are always tuple with : (num_nodes, shape of state....)
        n = self.parameters['num_nodes']
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('y1', shape=n)
        self.add_input('param_a', shape=(1, 2))
        self.add_input('param_b', shape=n)
        self.add_output('dy_dt', shape=(n, 2, 1))
        self.add_output('dy1_dt', shape=n)

        self.declare_derivatives(of='dy_dt', wrt='y')
        self.declare_derivatives(of='dy_dt', wrt='param_a')
        self.declare_derivatives(of='dy_dt', wrt='param_b')
        self.declare_derivatives(of='dy1_dt', wrt='y1')

    def compute(self, inputs, outputs):
        # Compute state for all (i, state) nodes where i = 0, ... n
        n = self.parameters['num_nodes']
        for i in range(n):
            outputs['dy_dt'][i] = np.array([[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [
                                           -0.1, inputs['param_b'][i]]]).dot(inputs['y'][i])
        outputs['dy1_dt'] = -0.1*(inputs['y1']*inputs['y1'])

    def compute_derivatives(self, inputs, partials):
         # Compute partials for all (i, state) nodes where i = 0, ... n
        n = self.parameters['num_nodes']
        for i in range(n):
            partials['dy_dt', 'y'][2*i:2*(i+1), 2*i:2*(i+1)] = np.array(
                [[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [-0.1, inputs['param_b'][i]]])
            partials['dy_dt', 'param_a'][2*i:2*(i+1), :] = np.array(
                [[inputs['y'][i][0, 0], inputs['y'][i][1, 0]], [0., 0.]])
            partials['dy_dt', 'param_b'][2*i:2 *
                                         (i+1), i:(i+1)] = np.array([[0.], [inputs['y'][i][1, 0]]])
        partials['dy1_dt', 'y1'] = -0.2*np.diag((inputs['y1']))

#  ODE System Model Class


class ODESystemModel(csdl.Model):
    def initialize(self):
            # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Shapes are always tuple with : (num_nodes, shape of state....)
        n = self.parameters['num_nodes']
        self.create_input('y', shape=(n, 2, 1))
        self.create_input('y1', shape=n)
        self.declare_variable('param_a', shape=(1, 2))
        self.declare_variable('param_b', shape=n)

        self.add(ODESystemOP(num_nodes=n), 'ODE', promotes=['*'])
        self.declare_variable('dy_dt')
        self.declare_variable('dy1_dt')
