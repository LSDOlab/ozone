
from modopt import CSDLProblem
from modopt import SLSQP
import matplotlib.pyplot as plt
from ode_systems import ODESystemNative, ODESystemModel
from ozone.api import ODEProblem, NativeSystem
import csdl
import python_csdl_backend
import numpy as np

# ODE Model with CSDL:
# Same ODE Model as coupled problem. However, the four coefficients a,b,g,d are now csdl variables that can be connected from outside
"""
This example showcases the following:
- ability to define dynamic and static parameters to your ODE. This is
the same ODE Model as the 'coupled_ODE' example. However, the four coefficients a,b,g,d are now csdl variables that can be as CSDL variables.
"""


# ODE problem CLASS
class ODEProblemTest(ODEProblem):
    def setup(self):
        # Define field outputs, profile outputs, states, parameters, times
        # Outputs. coefficients for field outputs must be defined as an upstream variable
        self.add_field_output('field_output', state_name='x', coefficients_name='coefficients')

        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        self.add_parameter('a', dynamic=True, shape=(self.num_times))
        self.add_parameter('b', dynamic=True, shape=(self.num_times))
        self.add_parameter('g', dynamic=True, shape=(self.num_times))
        # If dynamic != True, it is a static parameter. i.e, the parameter used in the ODE is constant through time.
        # Therefore, the shape does not depend on the number of timesteps
        self.add_parameter('d')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0')
        self.add_state('x', 'dx_dt', initial_condition_name='x_0', output='x_solved', interp_guess=[0, 1])
        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.set_ode_system(ODESystemModel)
        # self.set_ode_system(ODESystemNative)

# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_times')

    def define(self):
        num_times = self.parameters['num_times']

        h_stepsize = 0.1

        # Create given inputs
        # Coefficients for field output
        coeffs = self.create_input('coefficients', np.ones(num_times)/(num_times))
        # Initial condition for state
        y_0 = self.create_input('y_0', 2.0)
        x_0 = self.create_input('x_0', 2.0)

        # Create parameter for parameters a,b,g,d
        a = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        b = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        g = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        d = 0.5  # static parameter
        for t in range(num_times):
            a[t] = 1.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
            b[t] = 0.5 + t/num_times/5.0  # dynamic parameter defined at every timestep
            g[t] = 2.0 + t/num_times/5.0  # dynamic parameter defined at every timestep

        # Add to csdl model which are fed into ODE Model
        ai = self.create_input('a', a)
        bi = self.create_input('b', b)
        gi = self.create_input('g', g)
        di = self.create_input('d', d)

        # Timestep vector
        h_vec = np.ones(num_times-1)*h_stepsize
        h = self.create_input('h', h_vec)

        # Create Model containing integrator
        ODEProblem = ODEProblemTest('GaussLegendre6', 'collocation', num_times, display='default', visualization='end')
        # ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times, display='default', visualization='None')

        self.add(ODEProblem.create_solver_model(), 'subgroup')

        fo = self.declare_variable('field_output')
        self.register_output('fo', fo*1.0)

        x_integrated = self.declare_variable('x_solved', shape=(num_times, 1))
        self.register_output('x_integrated', x_integrated*1.0)

        # When using the collocation method, the ODE is solved using an optimization problem
        # where the constraints define the integration.
        # Therefore, an objective is needed. A dummy objective is used for this example
        # but a different objective can be specified.
        dummy_input = self.create_input('dummy_input')
        self.register_output('dummy_objective', dummy_input*1.0)
        self.add_objective('dummy_objective')


# Simulator Object:
# sim = python_csdl_backend.Simulator(RunModel(num_times=10), mode='rev')
sim = python_csdl_backend.Simulator(RunModel(num_times=10), mode='rev')

sim.run()
# PLOT SPARSITY
# Uncomment for constraint jacobian
# sim.compute_total_derivatives()
# x = sim.constraint_jacobian()
# fig, ax = plt.subplots()
# ax.spy(x)
# plt.savefig("constraint_jac_sparsity.png", bbox_inches='tight')
# print('Constraint Jacobian Sparsity: ', 1.0 - (np.count_nonzero(x) / float(x.size)))
# plt.show()
# exit()

prob = CSDLProblem(
    problem_name='collocation_example',
    simulator=sim,
)

optimizer = SLSQP(prob)

# Solve your optimization problem
optimizer.solve()
optimizer.print_results()

# Visualize
# plt.plot(sim['x_solved'])
# plt.grid()
# plt.show()
# print(sim['stage__x'])
# print(sim['stage__dx_dt'])
print(sim['x_solved'])
# OLD: [2.         2.83602753 3.96361838 5.33822479 6.78398233 8.04428573 8.92496754 9.38554183 9.50297712 9.38839602] # 10 values
