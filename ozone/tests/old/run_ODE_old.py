

# ODE Model with CSDL:
# Same ODE Model as coupled problem. However, the four coefficients a,b,g,d are now csdl variables that can be connected from outside
from ozone.api import ODEProblem
import csdl
# import openmdao.api as om
from ozone.api import Wrap, NativeSystem
import python_csdl_backend
import numpy as np


def run_ode(settings_dict):
    from ozone.api import ODEProblem

    numerical_method = settings_dict['num_method']
    approach_test = settings_dict['approach']
    system_type = settings_dict['system']
    fwd_solver = settings_dict['fwd_solver']
    jvp_solver = settings_dict['jvp_solver']

    # ODE problem CLASS
    class ODEProblemTest(ODEProblem):
        def setup(self):
            # Define field outputs, profile outputs, states, parameters, times

            # Outputs. coefficients for field outputs must be defined as an upstream variable
            self.add_field_output('field_output_x', state_name='x', coefficients_name='coefficients')
            self.add_field_output('field_output_y', state_name='y', coefficients_name='coefficients')
            self.add_field_output('field_output_z', state_name='z', coefficients_name='coefficients')

            self.add_parameter('a', dynamic=True, shape=(num))
            self.add_parameter('b', dynamic=True, shape=(num))
            self.add_parameter('g', dynamic=True, shape=(num))
            self.add_parameter('d')

            # Inputs names correspond to respective upstream CSDL variables
            self.add_state('y', 'dy_dt', initial_condition_name='y_0')
            self.add_state('x', 'dx_dt', initial_condition_name='x_0')
            self.add_state('z', 'dz_dt', initial_condition_name='z_0', shape=2)
            self.add_times(step_vector='h')

            # Define ODE and Profile Output systems (Either CSDL Model or Native System)
            if system_type == 'CSDL':
                self.ode_system = Wrap(ODESystemModel)
            elif system_type == 'NSstd':
                self.ode_system = ODESystemNative()
            elif system_type == 'NSspr':
                self.ode_system = ODESystemNativeSparse()

    class RunModel(csdl.Model):
        def define(self):

            h_stepsize = 0.05

            # Create given inputs
            # Coefficients for field output
            self.create_input('coefficients', np.ones(num+1)/(num+1))
            # Initial condition for state
            self.create_input('y_0', 2.0)
            self.create_input('x_0', 2.0)
            self.create_input('z_0', [2.0, -2.0])

            # Create parameter for parameters a,b,g,d
            a = np.zeros((num, 1))  # dynamic parameter defined at every timestep
            b = np.zeros((num, 1))  # dynamic parameter defined at every timestep
            g = np.zeros((num, 1))  # dynamic parameter defined at every timestep
            d = 0.5  # static parameter
            for t in range(num):
                a[t] = 1.0 + t/num/5.0  # dynamic parameter defined at every timestep
                b[t] = 0.5 + t/num/5.0  # dynamic parameter defined at every timestep
                g[t] = 2.0 + t/num/5.0  # dynamic parameter defined at every timestep

            # Add to csdl model which are fed into ODE Model
            self.create_input('a', a)
            self.create_input('b', b)
            self.create_input('g', g)
            self.create_input('d', d)

            # Timestep vector
            h_vec = np.ones(num)*h_stepsize
            self.create_input('h', h_vec)

            # Create Model containing integrator
            self.add(ODEProblem.create_solver_model(), 'subgroup')

            fox = self.declare_variable('field_output_x')
            foy = self.declare_variable('field_output_y')
            foz = self.declare_variable('field_output_z', shape=(2,))

            self.register_output('total', fox + foy + foz[0] + foz[1])

    # ODEProblem_instance
    num = 10

    # Integration approach: RK4 Timeamarching
    ODEProblem = ODEProblemTest(
        numerical_method,
        approach_test,
        num,
        display='default',
        visualization='None',
        implicit_solver_fwd=fwd_solver,
        implicit_solver_jvp=jvp_solver)

    # Simulator Object:
    sim = python_csdl_backend.Simulator(RunModel(), mode='rev')
    sim.prob.run_model()
    val = sim['total']
    print(val)
    # sim.visualize_implementation()

    # # Checktotals
    # print(sim.prob['field_output'])
    derivative_checks = sim.prob.check_totals(of=['total'], wrt=[
        'd', 'a', 'y_0', 'x_0'], compact_print=True)

    return_dict = {'output': val, 'derivative_checks': derivative_checks}
    return return_dict


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.create_input('y', shape=n)
        x = self.create_input('x', shape=n)
        z = self.create_input('z', shape=(n, 2))
        dz_dt = self.create_output('dz_dt', shape=(n, 2))

        # Paramters are now inputs
        a = self.create_input('a', shape=(n))
        b = self.create_input('b', shape=(n))
        g = self.create_input('g', shape=(n))
        d = self.create_input('d')

        # Predator Prey ODE:
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y - csdl.expand(d, n)*x
        dz_dt[:, 0] = -1/10.0*csdl.expand(y, (n, 1), 'i->ij') + z[:, 0]
        dz_dt[:, 1] = -1/20.0*csdl.expand(y, (n, 1), 'i->ij') + z[:, 1]

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)
        # self.register_output('dz_dt', dz_dt)


class ODESystemNative(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # Need to have ODE shapes similar as first example
        n = self.num_nodes
        self.add_input('y', shape=n)
        self.add_input('x', shape=n)
        self.add_input('z', shape=(n, 2))
        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)
        self.add_output('dz_dt', shape=(n, 2))

        self.add_input('a', shape=n)
        self.add_input('b', shape=n)
        self.add_input('g', shape=n)
        self.add_input('d')

        self.declare_partial_properties('dy_dt', 'g', empty=True)
        self.declare_partial_properties('dy_dt', 'd', empty=True)
        self.declare_partial_properties('dx_dt', 'a', empty=True)
        self.declare_partial_properties('dx_dt', 'b', empty=True)
        self.declare_partial_properties('dy_dt', 'z', empty=True)
        self.declare_partial_properties('dx_dt', 'z', empty=True)

        self.declare_partial_properties('dz_dt', 'g', empty=True)
        self.declare_partial_properties('dz_dt', 'd', empty=True)
        self.declare_partial_properties('dz_dt', 'a', empty=True)
        self.declare_partial_properties('dz_dt', 'b', empty=True)
        self.declare_partial_properties('dz_dt', 'x', empty=True)
    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO

    def compute(self, inputs, outputs):
        n = self.num_nodes
        a = inputs['a']
        b = inputs['b']
        g = inputs['g']
        d = inputs['d']

        # Outputs
        outputs['dy_dt'] = np.multiply(a, inputs['y']) - np.multiply(b, np.multiply(inputs['y'], inputs['x']))
        outputs['dx_dt'] = np.multiply(g, np.multiply(inputs['y'], inputs['x'])) - d*inputs['x']
        outputs['dz_dt'] = np.zeros((n, 2))
        outputs['dz_dt'][:, 0] = -inputs['y']/10.0 + inputs['z'][:, 0]
        outputs['dz_dt'][:, 1] = -inputs['y']/20.0 + inputs['z'][:, 1]

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        a = inputs['a']
        b = inputs['b']
        g = inputs['g']
        d = inputs['d']

        # The partials to compute.
        partials['dy_dt']['y'] = np.diag(a - b*inputs['x'])
        partials['dy_dt']['x'] = np.diag(- b*inputs['y'])
        partials['dx_dt']['y'] = np.diag(g*inputs['x'])
        partials['dx_dt']['x'] = np.diag(g*inputs['y']-d)

        partials['dy_dt']['a'] = np.diag(inputs['y'])
        partials['dy_dt']['b'] = np.diag(-np.multiply(inputs['y'], inputs['x']))
        partials['dx_dt']['d'] = np.array(-inputs['x'])
        partials['dx_dt']['g'] = np.diag(np.multiply(inputs['y'], inputs['x']))

        partials['dz_dt']['z'] = np.diag(np.ones(2*n))

        partials['dz_dt']['y'] = np.zeros((2*n, n))
        for i in range(n):
            partials['dz_dt']['y'][2*i:2*i+2, i:i+1] = np.array([[-1/10], [-1/20]])

        # The structure of partials has the following for n = self/num_nodes =  4:
        # d(dy_dt)/dy =
        # [d(dy_dt1)/dy1  0               0               0            ]
        # [0              d(dy_dt2)/dy2   0               0            ]
        # [0              0               d(dy_dt2)/dy2   0            ]
        # [0              0               0               d(dy_dt2)/dy2]
        # Hence the diagonal


class ODESystemNativeSparse(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # Need to have ODE shapes similar as first example
        n = self.num_nodes
        self.add_input('y', shape=n)
        self.add_input('x', shape=n)
        self.add_input('z', shape=(n, 2))
        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)
        self.add_output('dz_dt', shape=(n, 2))

        self.add_input('a', shape=n)
        self.add_input('b', shape=n)
        self.add_input('g', shape=n)
        self.add_input('d')

        # x,y derivatives:
        self.declare_partial_properties('dy_dt', 'g', empty=True)
        self.declare_partial_properties('dy_dt', 'd', empty=True)
        self.declare_partial_properties('dx_dt', 'a', empty=True)
        self.declare_partial_properties('dx_dt', 'b', empty=True)
        self.declare_partial_properties('dy_dt', 'z', empty=True)
        self.declare_partial_properties('dx_dt', 'z', empty=True)

        # Sparse:
        row_col = np.arange(n)
        self.declare_partial_properties('dy_dt', 'y', rows=row_col, cols=row_col)
        self.declare_partial_properties('dy_dt', 'x', rows=row_col, cols=row_col)
        self.declare_partial_properties('dy_dt', 'a', rows=row_col, cols=row_col)
        self.declare_partial_properties('dy_dt', 'b', rows=row_col, cols=row_col)
        self.declare_partial_properties('dx_dt', 'y', rows=row_col, cols=row_col)
        self.declare_partial_properties('dx_dt', 'x', rows=row_col, cols=row_col)
        self.declare_partial_properties('dx_dt', 'g', rows=row_col, cols=row_col)

        # z derivatives:
        self.declare_partial_properties('dz_dt', 'g', empty=True)
        self.declare_partial_properties('dz_dt', 'd', empty=True)
        self.declare_partial_properties('dz_dt', 'a', empty=True)
        self.declare_partial_properties('dz_dt', 'b', empty=True)
        self.declare_partial_properties('dz_dt', 'x', empty=True)
        row_col = np.arange(2*n)
        vals = np.ones(2*n)
        self.declare_partial_properties('dz_dt', 'z', rows=row_col, cols=row_col, vals=vals)

    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO

    def compute(self, inputs, outputs):
        n = self.num_nodes
        a = inputs['a']
        b = inputs['b']
        g = inputs['g']
        d = inputs['d']

        # Outputs
        outputs['dy_dt'] = np.multiply(a, inputs['y']) - np.multiply(b, np.multiply(inputs['y'], inputs['x']))
        outputs['dx_dt'] = np.multiply(g, np.multiply(inputs['y'], inputs['x'])) - d*inputs['x']
        outputs['dz_dt'] = np.zeros((n, 2))
        outputs['dz_dt'][:, 0] = -inputs['y']/10.0 + inputs['z'][:, 0]
        outputs['dz_dt'][:, 1] = -inputs['y']/20.0 + inputs['z'][:, 1]

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        a = inputs['a']
        b = inputs['b']
        g = inputs['g']
        d = inputs['d']

        # The partials to compute.
        partials['dy_dt']['y'] = (a - b*inputs['x'])
        partials['dy_dt']['x'] = (- b*inputs['y'])
        partials['dx_dt']['y'] = (g*inputs['x'])
        partials['dx_dt']['x'] = (g*inputs['y']-d)

        partials['dy_dt']['a'] = (inputs['y'])
        partials['dy_dt']['b'] = (-np.multiply(inputs['y'], inputs['x']))
        partials['dx_dt']['d'] = np.array(-inputs['x'])
        partials['dx_dt']['g'] = (np.multiply(inputs['y'], inputs['x']))

        partials['dz_dt']['y'] = np.zeros((2*n, n))
        for i in range(n):
            partials['dz_dt']['y'][2*i:2*i+2, i:i+1] = np.array([[-1/10], [-1/20]])

        # The structure of partials has the following for n = self/num_nodes =  4:
        # d(dy_dt)/dy =
        # [d(dy_dt1)/dy1  0               0               0            ]
        # [0              d(dy_dt2)/dy2   0               0            ]
        # [0              0               d(dy_dt2)/dy2   0            ]
        # [0              0               0               d(dy_dt2)/dy2]
        # Hence the diagonal
