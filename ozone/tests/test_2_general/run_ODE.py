
import matplotlib.pyplot as plt
# import openmdao.api as om
from ozone.api import NativeSystem
import csdl
import python_csdl_backend
import numpy as np
from ozone.tests.test_2_general.run_ODE_systems import ODESystemNative, ODESystemCSDL, POSystem


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
            self.add_field_output('field_output_z', state_name='z', coefficients_name='coefficients')
            self.add_field_output('field_output_y', state_name='y', coefficients_name='coefficients')

            self.add_profile_output('profile_output_x')
            self.add_profile_output('profile_output_z')
            self.add_profile_output('profile_output_y', shape=(2, 2))

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
            self.add_state('y', 'dy_dt', initial_condition_name='y_0')
            self.add_state('x', 'dx_dt', initial_condition_name='x_0')
            self.add_state('z', 'dz_dt', initial_condition_name='z_0', shape=(2, 2), output='z_integrated')
            self.add_times(step_vector='h')

            # Define ODE and Profile Output systems (Either CSDL Model or Native System)

            # ODE
            if system_type == 'CSDL':
                self.set_ode_system(ODESystemCSDL)  # CSDL
            elif system_type == 'NSstd':
                self.set_ode_system(ODESystemNative)  # NATIVE

            # Profile
            self.set_profile_system(POSystem)
            # self.set_profile_system(Wrap(POSystemModel))

    # The CSDL Model containing the ODE integrator

    class RunModel(csdl.Model):
        def initialize(self):
            self.parameters.declare('num_timesteps')

        def define(self):
            num_times = self.parameters['num_timesteps']

            h_stepsize = 0.001

            # Create given inputs
            # Coefficients for field output
            self.create_input('coefficients', np.ones(num_times)/(num_times))
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
            h_vec = np.ones(num_times-1)*h_stepsize
            self.create_input('h', h_vec)

            # Create Model containing integrator

            self.add(ODEProblem.create_solver_model(), 'subgroup')

            foy = self.declare_variable('field_output_y')
            pox = self.declare_variable('profile_output_x', shape=(num_times, ))
            poz = self.declare_variable('profile_output_z', shape=(num_times, ))
            poy = self.declare_variable('profile_output_y', shape=(num_times, 2, 2))
            z_int = self.declare_variable('z_integrated', shape=(num_times, 2, 2))
            temp = csdl.reshape(z_int[-1, 0, 1], 1)
            # print(temp.shape)

            self.register_output('total', pox[-1]+poz[-1]+foy[0]+temp/2.0)
            self.register_output('total2', csdl.pnorm(poy[-1, :, :] + poy[0, :, :]))

            if approach_test == 'collocation':
                dummy_input = self.create_input('dummy_input')
                self.register_output('dummy_out', dummy_input*1.0)
                self.add_objective('dummy_out')

    # Simulator Object:
    nt = settings_dict['numtimes']

    ODEProblem = ODEProblemTest(
        numerical_method,
        approach_test,
        nt,
        display='default',
        visualization='None',
        implicit_solver_fwd=fwd_solver,
        implicit_solver_jvp=jvp_solver)

    sim = python_csdl_backend.Simulator(RunModel(num_timesteps=nt), mode='rev')
    sim.run()

    if approach_test == 'collocation':
        from modopt.scipy_library import SLSQP
        from modopt.csdl_library import CSDLProblem
        prob = CSDLProblem(
            problem_name='test_2',
            simulator=sim,
        )

        optimizer = SLSQP(prob)

        # Solve your optimization problem
        optimizer.solve()

    val = sim['total']
    print('total: ', val)
    val2 = sim['total2']
    print('total2: ', val2)
    val = {
        'total': np.array(val),
        'total2':  np.array(val2),
    }
    # exit()
    # total:  [2.8050334]
    # total2:  22.27925077337252

    derivative_checks = sim.compute_totals(of=['total', 'total2'], wrt=['a', 'x_0', 'h', 'z_0', 'e'])
    # sim.check_totals(of=['total','total2'], wrt=['a', 'x_0', 'h', 'z_0', 'e'])
    # exit()
    for key in derivative_checks:
        print('derivative norm:', key, np.linalg.norm(derivative_checks[key]))

    return_dict = {'output': val, 'derivative_checks': derivative_checks}

    if settings_dict['benchmark']:

        # sim.prob.check_totals(of=['total'], wrt=['x_0', 'h', 'a', 'd'], compact_print=True)

        bench = True
        bench = False
        if bench:
            import pickle
            import time

            num_timesteps = nt
            num_time = 50
            num_time_int = 50
            time_list = []
            for i in range(num_time):
                t_start = time.time()
                sim.compute_totals(of=['total'], wrt=['a', 'z_0', 'h', 'd'])
                time_list.append(time.time() - t_start)
            time_list_int = []
            for i in range(num_time_int):
                t_start = time.time()
                sim.run_model()
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

    return return_dict


if __name__ == '__main__':

    settings_dict = {
        'approach': 'time-marching',
        'system': 'NSstd',
        'fwd_solver': 'iterative',
        'jvp_solver': 'iterative',
        'num_method': 'Trapezoidal',
        'benchmark': True,
        'numtimes': 30
    }
    settings_dict['num_method'] = 'RK4'  # [4.74133715]
    # settings_dict['approach'] = 'time-marching checkpointing'

    run_ode(settings_dict)
