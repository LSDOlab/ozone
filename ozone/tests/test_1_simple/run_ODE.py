
import matplotlib.pyplot as plt
# import openmdao.api as om
from ozone.api import NativeSystem, ODEProblem
import csdl
import python_csdl_backend
import numpy as np
from ozone.tests.test_1_simple.run_ODE_systems import ODESystemNative, ODESystemCSDL


def run_ode(settings_dict):

    from ozone.api import ODEProblem

    numerical_method = settings_dict['num_method']
    approach_test = settings_dict['approach']
    system_type = settings_dict['system']
    fwd_solver = settings_dict['fwd_solver']
    jvp_solver = settings_dict['jvp_solver']

    # The CSDL Model containing the ODE integrator

    class RunModel(csdl.Model):
        def initialize(self):
            self.parameters.declare('num_timesteps')

        def define(self):
            num_times = self.parameters['num_timesteps']

            h_stepsize = 0.01

            # Initial condition for state
            self.create_input('y_0', 0.5)
            h_vec = np.ones(num_times-1)*h_stepsize
            self.create_input('h', h_vec)

            # Create Model containing integrator
            ode_problem = ODEProblem(
                numerical_method,
                approach_test,
                nt,
                display='default',
                visualization='None',
                implicit_solver_fwd=fwd_solver,
                implicit_solver_jvp=jvp_solver,
            )

            ode_problem.add_state('y', 'dy_dt', initial_condition_name='y_0', output='y_integrated')
            ode_problem.add_times(step_vector='h')

            # ODE
            if system_type == 'CSDL':
                ode_system = ODESystemCSDL  # CSDL
            elif system_type == 'NSstd':
                ode_system = ODESystemNative  # NATIVE

            ode_problem.set_ode_system(ode_system)

            self.add(ode_problem.create_solver_model())

            y_out = self.declare_variable('y_integrated', shape=(nt,))

            self.register_output('y_out', csdl.sum(y_out))

            if approach_test == 'collocation':
                dummy_input = self.create_input('dummy_input')
                self.register_output('dummy_out', dummy_input*1.0)
                self.add_objective('dummy_out')

    # Simulator Object:
    nt = settings_dict['numtimes']

    model = RunModel(num_timesteps=nt)
    rep = csdl.GraphRepresentation(model)
    sim = python_csdl_backend.Simulator(rep, mode='rev')
    sim.run()

    if approach_test == 'collocation':
        from modopt.scipy_library import SLSQP
        from modopt.csdl_library import CSDLProblem
        prob = CSDLProblem(
            problem_name='test_3',
            simulator=sim,
        )

        optimizer = SLSQP(prob)

        # Solve your optimization problem
        optimizer.solve()

    val = sim['y_out']
    print('y_out: ', val)

    derivative_checks = sim.compute_totals(of=['y_out'], wrt=['y_0', 'h'])
    for key in derivative_checks:
        print('derivative norm:', key, np.linalg.norm(derivative_checks[key]))
    # sim.check_partials(compact_print=1)
    # sim.check_totals(of=['stage__y', 'state__dy_dt'], wrt='y_0')

    # exit()
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
