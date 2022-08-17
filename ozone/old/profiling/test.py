import pickle
import pstats
import cProfile
import matplotlib.pyplot as plt
from AttitudeNative import AttitudeNS
from python_csdl_backend import Simulator
from ozone.api import ODEProblem, Wrap
from csdl import Model
import csdl
import numpy as np


class Attitude(Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        # self.parameters.declare('K')

    def define(self):
        n = self.parameters['num_nodes']

        # K = self.parameters['K']
        K = self.create_input('K', shape=(3,))

        Omega = self.create_input('Omega', shape=n)
        t0 = self.create_input('t0', shape=n)
        t1 = self.create_input('t1', shape=n)
        t2 = self.create_input('t2', shape=n)

        omega0 = self.create_input('omega0', shape=n)
        omega1 = self.create_input('omega1', shape=n)
        omega2 = self.create_input('omega2', shape=n)

        C00 = self.create_input('C00', shape=n)
        C01 = self.create_input('C01', shape=n)
        C02 = self.create_input('C02', shape=n)
        C10 = self.create_input('C10', shape=n)
        C11 = self.create_input('C11', shape=n)
        C12 = self.create_input('C12', shape=n)
        C20 = self.create_input('C20', shape=n)
        C21 = self.create_input('C21', shape=n)
        C22 = self.create_input('C22', shape=n)

        # Compute angular acceleration
        # p = Omega**2
        do0_dt = csdl.expand(K[0], n)*(omega1 * omega2 - 3 *
                                       Omega**2 * C01 * C02 + t0)
        do1_dt = csdl.expand(K[1], n)*(omega2 * omega0 - 3 *
                                       Omega**2 * C02 * C00 + t1)
        do2_dt = csdl.expand(K[2], n)*(omega0 * omega1 - 3 *
                                       Omega**2 * C00 * C01 + t2)

        # Update direction cosine matrix for body rotating in frame
        # fixed in orbit
        dC00_dt = C01 * omega2 - C02 * omega1 + Omega * (C02 * C21 - C01 * C22)
        dC01_dt = C02 * omega0 - C00 * omega2 + Omega * (C00 * C22 - C02 * C20)
        dC02_dt = C00 * omega1 - C01 * omega0 + Omega * (C01 * C20 - C00 * C21)
        dC10_dt = C11 * omega2 - C12 * omega1 + Omega * (C12 * C21 - C11 * C22)
        dC11_dt = C12 * omega0 - C10 * omega2 + Omega * (C10 * C22 - C12 * C20)
        dC12_dt = C10 * omega1 - C11 * omega0 + Omega * (C11 * C20 - C10 * C21)
        dC20_dt = C21 * omega2 - C22 * omega1
        dC21_dt = C22 * omega0 - C20 * omega2
        dC22_dt = C20 * omega1 - C21 * omega0

        self.register_output('do0_dt', do0_dt)
        self.register_output('do1_dt', do1_dt)
        self.register_output('do2_dt', do2_dt)
        self.register_output('dC00_dt', dC00_dt)
        self.register_output('dC01_dt', dC01_dt)
        self.register_output('dC02_dt', dC02_dt)
        self.register_output('dC10_dt', dC10_dt)
        self.register_output('dC11_dt', dC11_dt)
        self.register_output('dC12_dt', dC12_dt)
        self.register_output('dC20_dt', dC20_dt)
        self.register_output('dC21_dt', dC21_dt)
        self.register_output('dC22_dt', dC22_dt)

        # self.print_var(do0_dt)


class P(ODEProblem):
    def setup(self):
        # Define field outputs, profile outputs, states, parameters,
        # times
        self.add_profile_output('omega0', state_name='omega0', shape=(1,))
        self.add_profile_output('omega1', state_name='omega1', shape=(1,))
        self.add_profile_output('omega2', state_name='omega2', shape=(1,))
        self.add_profile_output('C00', state_name='C00', shape=(1,))
        self.add_profile_output('C01', state_name='C01', shape=(1,))
        self.add_profile_output('C02', state_name='C02', shape=(1,))
        self.add_profile_output('C10', state_name='C10', shape=(1,))
        self.add_profile_output('C11', state_name='C11', shape=(1,))
        self.add_profile_output('C12', state_name='C12', shape=(1,))
        self.add_profile_output('C20', state_name='C20', shape=(1,))
        self.add_profile_output('C21', state_name='C21', shape=(1,))
        self.add_profile_output('C22', state_name='C22', shape=(1,))

        # TODO: ...

        # add states:
        # TODO: IC names
        self.add_state('omega0', 'do0_dt', initial_condition_name='omega0_0')
        self.add_state('omega1', 'do1_dt', initial_condition_name='omega1_0')
        self.add_state('omega2', 'do2_dt', initial_condition_name='omega2_0')
        self.add_state('C00', 'dC00_dt', initial_condition_name='C00_0')
        self.add_state('C01', 'dC01_dt', initial_condition_name='C01_0')
        self.add_state('C02', 'dC02_dt', initial_condition_name='C02_0')
        self.add_state('C10', 'dC10_dt', initial_condition_name='C10_0')
        self.add_state('C11', 'dC11_dt', initial_condition_name='C11_0')
        self.add_state('C12', 'dC12_dt', initial_condition_name='C12_0')
        self.add_state('C20', 'dC20_dt', initial_condition_name='C20_0')
        self.add_state('C21', 'dC21_dt', initial_condition_name='C21_0')
        self.add_state('C22', 'dC22_dt', initial_condition_name='C22_0')

        # TODO: ...

        self.add_parameter('t0', dynamic=True, shape=(nt))
        self.add_parameter('t1', dynamic=True, shape=(nt))
        self.add_parameter('t2', dynamic=True, shape=(nt))
        self.add_parameter('Omega', dynamic=True, shape=(nt))
        self.add_parameter('K', shape=(3,))

        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.ode_system = Wrap(Attitude)
        self.profile_outputs_system = Wrap(ProfileSystemModel)

        self.ode_system = AttitudeNS()


class ProfileSystemModel(Model):
    """
    takes in a  state at a time step and then it outputs like your
    output that you define
    processing the states as an output

    """

    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Shapes are always tuple with : (num_nodes, shape of state....)
        n = self.parameters['num_nodes']
        omega0 = self.create_input('omega0', shape=n)
        omega1 = self.create_input('omega1', shape=n)
        omega2 = self.create_input('omega2', shape=n)
        C00 = self.create_input('C00', shape=n)
        C01 = self.create_input('C01', shape=n)
        C02 = self.create_input('C02', shape=n)
        C10 = self.create_input('C10', shape=n)
        C11 = self.create_input('C11', shape=n)
        C12 = self.create_input('C12', shape=n)
        C20 = self.create_input('C20', shape=n)
        C21 = self.create_input('C21', shape=n)
        C22 = self.create_input('C22', shape=n)

        # TODO: ...
nt = 100


class RunModel(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('K')

    def define(self):
        n = self.parameters['num_times']
        K_param = self.parameters['K']
        h_stepsize = self.parameters['step_size']

        # Create given inputs
        # Initial condition for state
        self.create_input('omega0_0', 2.0)
        self.create_input('omega1_0', 3.0)
        self.create_input('omega2_0', 4.0)
        self.create_input('C00_0', 0.99)
        self.create_input('C01_0', -0.0868)
        self.create_input('C02_0', 0.0872)
        self.create_input('C20_0', -0.0789)
        self.create_input('C21_0', 0.0944)
        self.create_input('C22_0', 0.9924)
        self.create_input('C10_0', 0.9924)
        self.create_input('C11_0',  0.99174311)
        self.create_input('C12_0', -0.08682396)

        # Add to csdl model which are fed into ODE Model
        t0 = np.ones((n))*1
        t1 = np.ones((n))*0.1
        t2 = np.ones((n))*1.1

        t0 = self.create_input('t0', shape=n)
        t1 = self.create_input('t1', shape=n)
        t2 = self.create_input('t2', shape=n)
        Omega = self.create_input('Omega', shape=n)
        K = self.create_input('K', val=K_param)

        # Timestep vector
        h_vec = np.ones(nt) * h_stepsize
        self.create_input('h', h_vec)

        # Create Model containing integrator
        # self.add(P('RK4',
        #            'time-marching',
        #            num_times=n,
        #            visualization='None',
        #            implicit_solver_fwd='direct',
        #            implicit_solver_jvp='direct').create_solver_model(), name='integrator')

        self.add(P('RK4',
                   'time-marching',
                   num_times=n,
                   visualization='None').create_solver_model(), name='integrator')

        omega0 = self.declare_variable('omega0', shape=(n+1, 1))
        self.register_output('out', csdl.sum(omega0))


K1 = -0.5
K2 = 0.9
K3 = -(K1 + K2) / (1 + K1 * K2)
K_temp = [K1, K2, K3]

sim = Simulator(RunModel(num_times=nt, step_size=0.05,
                         K=K_temp), mode='rev')
sim.run()
# sim.check_partials(compact_print=True, method='fd')
# print(sim['out'])
# out_list = ['Omega', 't0', 't1',
# 't2', 'omega0', 'omega1', 'omega2', 'C00', 'C01', 'C02', 'C10', 'C11', 'C12', 'C20', 'C21', 'C22']

sim.prob.check_totals(of=['out'],  wrt=['omega2_0'], compact_print=True)

# profiler = cProfile.Profile()
# profiler.enable()
# sim.prob.compute_totals(of=['out'],  wrt=['omega2_0'])
# # sim.run()

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()
# profiler.dump_stats('output')


# with open('lineprofiler', 'rb') as f:
#     data = pickle.load(f)
# print(data)


# exit()

# state = sim['angular_velocity_orientation']
# omega = state[:3, :]

# # NOTE: can't get precession or roll from only first and third row of C
# C0 = np.array(state[3:6, :])
# C1 = np.array(state[6:9, :])
# C2 = np.array(state[9:, :])
# precession = 180 / np.pi * np.arctan2(C2[0, :], -C2[1, :])
# nutation = 180 / np.pi * np.arccos(C2[2, :])
# spin = 180 / np.pi * np.arctan2(C0[2, :], C1[2, :])
# roll = 180 / np.pi * np.arctan2(-C2[1, :], C2[2, :])
# pitch = 180 / np.pi * np.arcsin(C2[0, :])
# yaw = 180 / np.pi * np.arctan2(-C1[0, :], C0[0, :])

# # plt.plot(np.unwrap(precession, 180.))
# plt.plot(nutation)
# # plt.plot(np.unwrap(spin, 180.))
# # plt.plot(roll)
# # plt.plot(pitch)
# # plt.plot(yaw)
# # plt.plot(omega.T)
# # plt.plot(C0.T)
# # plt.plot(C1.T)
# # plt.plot(C2.T)
plt.show()
