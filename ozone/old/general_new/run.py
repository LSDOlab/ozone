
import matplotlib.pyplot as plt
# import openmdao.api as om
from systems import ODESystemNative, ODESystemCSDL, POSystemNS
from ozone.api import ODEProblem, Wrap, NativeSystem
import csdl
import python_csdl_backend
import numpy as np


# ODE problem CLASS
class ODEProblemTest(ODEProblem):
    def setup(self):
        # Define field outputs, profile outputs, states, parameters, times

        # Outputs. coefficients for field outputs must be defined as an upstream variable
        self.add_field_output('field_output_z', state_name='z', coefficients_name='coefficients')
        self.add_field_output('field_output_y', state_name='y', coefficients_name='coefficients')

        self.add_profile_output('profile_output_x')
        self.add_profile_output('profile_output_z')

        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        self.add_parameter('a', dynamic=True, shape=(self.num_times))
        self.add_parameter('b', dynamic=True, shape=(self.num_times))
        self.add_parameter('g', dynamic=True, shape=(self.num_times))
        self.add_parameter('e', dynamic=True, shape=(self.num_times, 2, 2))
        # If dynamic != True, it is a static parameter. i.e, the parameter used in the ODE is constant through time.
        # Therefore, the shape does not depend on the number of timesteps
        self.add_parameter('d')

        # Inputs names correspond to respective upstream CSDL variables
        self.add_state('y', 'dy_dt', initial_condition_name='y_0', output='y_integrated')
        self.add_state('x', 'dx_dt', initial_condition_name='x_0')
        self.add_state('z', 'dz_dt', initial_condition_name='z_0', shape=(2, 2), output='z_integrated')

        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)

        # ODE
        # self.set_ode_system(ODESystemNative())     # NATIVE
        self.set_ode_system(Wrap(ODESystemCSDL))  # CSDL

        # Profile
        self.set_profile_system(POSystemNS())
        # self.set_profile_system(Wrap(POSystemModel))

# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_timesteps')

    def define(self):
        num_times = self.parameters['num_timesteps']

        h_stepsize = 0.05

        # Create given inputs
        # Coefficients for field output
        self.create_input('coefficients', np.ones(num_times+1)/(num_times+1))
        # Initial condition for state
        self.create_input('y_0', 2.0)
        self.create_input('x_0', 2.0)
        self.create_input('z_0', np.array([[2.0, 1.0], [-1.0, -3.0]]))

        # Create parameter for parameters a,b,g,d
        a = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        b = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        g = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
        e = np.zeros((num_times, 2, 2))  # dynamic parameter defined at every timestep
        d = 0.5  # static parameter
        for t in range(num_times):
            a[t] = 1.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
            b[t] = 0.5 + t/num_times/5.0  # dynamic parameter defined at every timestep
            g[t] = 2.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
            e[t, :, :] = np.array([[0.3, 0.2], [0.1, -2.6]]) + t/num_times/5.0  # dynamic parameter defined at every timestep

        # Add to csdl model which are fed into ODE Model
        self.create_input('a', a)
        self.create_input('b', b)
        self.create_input('g', g)
        self.create_input('d', d)
        self.create_input('e', e)

        # Timestep vector
        h_vec = np.ones(num_times)*h_stepsize
        self.create_input('h', h_vec)

        # Create Model containing integrator
        # ODEProblem = ODEProblemTest('RK4', 'time-marching', num_times, display='default', visualization='none')
        # ODEProblem = ODEProblemTest('RK4', 'time-marching checkpointing', num_times, display='default', visualization='none')
        ODEProblem = ODEProblemTest('RK4', 'solver-based', num_times, display='default', visualization='none')

        self.add(ODEProblem.create_solver_model(), 'subgroup')

        foy = self.declare_variable('field_output_y')
        pox = self.declare_variable('profile_output_x', shape=(num_times+1, ))
        poz = self.declare_variable('profile_output_z', shape=(num_times+1, ))
        y_int = self.declare_variable('y_integrated', shape=(num_times+1, ))
        z_int = self.declare_variable('z_integrated', shape=(num_times+1, 2, 2))

        self.register_output('total', pox[-1]+poz[-1]+foy[0]+y_int[-1])


# Simulator Object:
nt = 10
sim = python_csdl_backend.Simulator(RunModel(num_timesteps=nt), mode='rev')
sim.prob.run_model()
# sim.visualize_implementation()

# # Checktotals
print(sim.prob['total'])  # should be [8.83217126]
# sim.prob.check_totals(of=['total'], wrt=['x_0', 'a', 'h'], compact_print=True)

plt.show()

bench = False
# bench = True
if bench:
    import pickle
    import time

    num_timesteps = nt
    num_time = 50
    num_time_int = 50
    time_list = []
    for i in range(num_time):
        t_start = time.time()
        sim.prob.compute_totals(of=['total'], wrt=['a', 'z_0', 'h'])
        time_list.append(time.time() - t_start)
    time_list_int = []
    for i in range(num_time_int):
        t_start = time.time()
        sim.prob.run_model()
        time_list_int.append(time.time() - t_start)

    print('-----------JVP------------')
    for t in time_list:
        print(t)
    print('-----------INT------------')
    for t in time_list_int:
        print(t)
    # print('average integration:', sum(time_list_int) / len(time_list_int))
    # print('average jvp        :', sum(time_list) / len(time_list))

    with open("time_int", "rb") as fp:   # Unpickling
        b_int = pickle.load(fp)
    with open("time_jvp", "rb") as fp:   # Unpickling
        b_jvp = pickle.load(fp)

    with open("time_int", "wb") as fp:  # Pickling
        pickle.dump(time_list_int, fp)
    with open("time_jvp", "wb") as fp:  # Pickling
        pickle.dump(time_list, fp)

    fig, ax = plt.subplots(2)
    fig.suptitle(f'ODE with {num_timesteps} timesteps')
    ax[0].plot(b_jvp)
    ax[0].plot(time_list)
    ax[0].legend(['before code change', 'after code change'])
    ax[0].set_title('jvp')
    ax[0].set_xlabel('run #')
    ax[0].set_ylabel('runtime (sec)')
    ax[0].set_ylim(bottom=0)

    ax[1].plot(b_int)
    ax[1].plot(time_list_int)
    ax[1].legend(['before code change', 'after code change'])
    ax[1].set_title('integration')
    ax[1].set_xlabel('run #')
    ax[1].set_ylabel('runtime (sec)')
    ax[1].set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()
