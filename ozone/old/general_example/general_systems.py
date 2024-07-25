# import openmdao.api as om
import numpy as np
from ozone.api import NativeSystem
import scipy.linalg as spln

# Declare systems for "run_general_ex.py"
# Two options: OpenMDAO components or Native System classes

# Option 1:
# -------------- OPENMDAO COMPONENTS --------------


class ODESystem(om.ExplicitComponent):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.options.declare('num_nodes')

    def setup(self):
        # Shapes are always tuple with : (num_nodes, shape of state....)
        n = self.options['num_nodes']
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('y1', shape=n)
        self.add_input('param_a', shape=(1, 2))
        self.add_input('param_b', shape=n)
        self.add_output('dy_dt', shape=(n, 2, 1))
        self.add_output('dy1_dt', shape=n)

    def setup_partials(self):
        self.declare_partials('dy_dt', 'y')
        self.declare_partials('dy_dt', 'param_a')
        self.declare_partials('dy_dt', 'param_b')
        self.declare_partials('dy1_dt', 'y1')

    def compute(self, inputs, outputs):
        # Compute state for all (i, state) nodes where i = 0, ... n
        n = self.options['num_nodes']
        for i in range(n):
            outputs['dy_dt'][i] = np.array([[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [
                                           -0.1, inputs['param_b'][i]]]).dot(inputs['y'][i])
        outputs['dy1_dt'] = -0.1*(inputs['y1']*inputs['y1'])

    def compute_partials(self, inputs, partials):
         # Compute partials for all (i, state) nodes where i = 0, ... n
        # for i in inputs:
        #     print(i, inputs[i])
        # print()
        n = self.options['num_nodes']
        for i in range(n):
            partials['dy_dt', 'y'][2*i:2*(i+1), 2*i:2*(i+1)] = np.array(
                [[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [-0.1, inputs['param_b'][i]]])
            partials['dy_dt', 'param_a'][2*i:2*(i+1), :] = np.array(
                [[inputs['y'][i][0, 0], inputs['y'][i][1, 0]], [0., 0.]])
            partials['dy_dt', 'param_b'][2*i:2 *
                                         (i+1), i:(i+1)] = np.array([[0.], [inputs['y'][i][1, 0]]])
        partials['dy1_dt', 'y1'] = -0.2*np.diag((inputs['y1']))


class ProfileOutputSystem(om.ExplicitComponent):
    # Computes f_profile that takes state as input and outputs array with specified shape
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.options.declare('num_nodes')

    def setup(self):
        # Shapes are always tuple with : (num_nodes, shape of profile output....)
        n = self.options['num_nodes']
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('y1', shape=(n))
        self.add_output('profile_output', shape=n)
        self.add_output('profile_output2', shape=n)

        # We can precompute partials if constant
        val_s = np.ones((2*n))*0.5
        rows_s = np.concatenate((np.arange(0, n, 1), np.arange(0, n, 1)))
        cols_s = np.concatenate((np.arange(0, 2*n, 2), np.arange(1, 2*n+1, 2)))
        self.declare_partials('profile_output', 'y',
                              val=val_s, rows=rows_s, cols=cols_s)
        val_s2 = np.ones(n)
        rows_s2 = np.arange(0, n, 1)
        cols_s2 = np.arange(0, n, 1)
        self.declare_partials('profile_output2', 'y1',
                              val=val_s2, rows=rows_s2, cols=cols_s2)

    def compute(self, inputs, outputs):
        # Compute partials for all (i, state) nodes where i = 0, ... n
        outputs['profile_output'] = (
            inputs['y'][:, 0, 0] + inputs['y'][:, 1, 0])/2.
        outputs['profile_output2'] = inputs['y1']


# Option 2:
# -------------- NATIVE SYSTEM CLASSES --------------
class ODESystemNative(NativeSystem):

    def setup(self):
        # Required every time for ODE systems or Profile Output systems
        n = self.num_nodes

        # Defining inputs, outputs and partials have similar syntax to OpenMDAO components
        # Shapes are always tuple with : (num_nodes, shape of state....)
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('y1', shape=n)
        self.add_input('param_a', shape=(1, 2))
        self.add_input('param_b', shape=n)
        self.add_output('dy_dt', shape=(n, 2, 1))
        self.add_output('dy1_dt', shape=n)

        # Declaring empty partials increases efficiency if variables are uncoupled
        self.declare_partial_properties('dy1_dt', 'param_a', empty=True)
        self.declare_partial_properties('dy1_dt', 'param_b', empty=True)
        self.declare_partial_properties('dy1_dt', 'y', empty=True)
        self.declare_partial_properties('dy_dt', 'y1', empty=True)
        # self.declare_partial_properties(
        #     'dy1_dt', 'y1', complex_step_directional=True)

    def compute(self, inputs, outputs):
        # Again, similar syntax to OpenMDAO components
        # Compute states for all (i, state) nodes where i = 0, ... n
        n = self.num_nodes
        outputs['dy_dt'] = np.zeros((n, 2, 1))
        for i in range(n):
            outputs['dy_dt'][i] = np.array([[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [
                                           -0.1, inputs['param_b'][i][0]]]).dot(inputs['y'][i])
        outputs['dy1_dt'] = -0.1*(inputs['y1']*inputs['y1'])
        # print(inputs)

    def compute_partials(self, inputs, partials):
        # Again, similar syntax to OpenMDAO components
        # Compute partials for all (i, state) nodes where i = 0, ... n
        n = self.num_nodes
        dydty = []
        dydtb = []

        partials['dy_dt']['param_a'] = np.zeros((n*2, 2))
        for i in range(n):
            partials['dy_dt']['param_a'][2*i:2*(i+1), :] = np.array(
                [[inputs['y'][i][0, 0], inputs['y'][i][1, 0]], [0., 0.]])
            dydtb.append(np.array([[0.], [inputs['y'][i][1, 0]]]))
            dydty.append(np.array(
                [[inputs['param_a'][0, 0], inputs['param_a'][0, 1]], [-0.1, inputs['param_b'][i][0]]]))

        partials['dy_dt']['y'] = spln.block_diag(*dydty)
        partials['dy_dt']['param_b'] = spln.block_diag(*dydtb)
        partials['dy1_dt']['y1'] = np.diag(-0.2*inputs['y1'])


class ProfileOutputSystemNative(NativeSystem):
    # Computes f_profile that takes state as input and outputs array with specified shape
    def setup(self):
        # Required every time for ODE systems or Profile Output systems
        n = self.num_nodes

        # Defining inputs, outputs and partials have similar syntax to OpenMDAO components
        # Shapes are always tuple with : (num_nodes, shape of profile outputs....)
        self.add_input('y', shape=(n, 2, 1))
        self.add_input('y1', shape=(n))
        self.add_output('profile_output', shape=n)
        self.add_output('profile_output2', shape=n)

        # We can precompute partials if constant
        val_s = np.ones((2*n))*0.5
        rows_s = np.concatenate((np.arange(0, n, 1), np.arange(0, n, 1)))
        cols_s = np.concatenate((np.arange(0, 2*n, 2), np.arange(1, 2*n+1, 2)))
        self.declare_partial_properties(
            'profile_output', 'y', val=val_s, rows=rows_s, cols=cols_s)
        val_s2 = np.ones(n)
        rows_s2 = np.arange(0, n)
        cols_s2 = np.arange(0, n)
        # print(val_s2)
        self.declare_partial_properties(
            'profile_output2', 'y1', val=val_s2, rows=rows_s2, cols=cols_s2)

        # Declaring empty partials increases efficiency if variables are uncoupled
        self.declare_partial_properties('profile_output2', 'y', empty=True)
        self.declare_partial_properties('profile_output', 'y1', empty=True)

    def compute(self, inputs, outputs):
        # Again, similar syntax to OpenMDAO components
        # Compute states for all (i, profile output) nodes where i = 0, ... n
        outputs['profile_output'] = (
            inputs['y'][:, 0, 0] + inputs['y'][:, 1, 0])/2.
        outputs['profile_output2'] = inputs['y1']
