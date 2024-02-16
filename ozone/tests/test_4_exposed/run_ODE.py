
import matplotlib.pyplot as plt
from ozone.api import NativeSystem, ODEProblem
import csdl
import python_csdl_backend
import numpy as np

class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']

        y = self.declare_variable('y', shape=n)
        x = self.declare_variable('x', shape=n)

        # Paramters are now inputs
        a = self.declare_variable('a', shape=(n))
        b = self.declare_variable('b', shape=(n))
        g = self.declare_variable('g', shape=(n))
        d = self.declare_variable('d')

        # Predator Prey ODE:
        iv1 = self.register_output('intermediate_variable_1', x*y) # intermediate variable exposed as output
        dy_dt = a*y - b*iv1
        dx_dt = g*iv1 - csdl.expand(d, n)*x


        iv2 = self.register_output('intermediate_variable_2', (dy_dt + y + x + a)**2.0) # intermediate variable exposed as output

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)

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

            h_stepsize = 0.005

            coeffs = self.create_input('coefficients', np.ones(num_times)/(num_times))
            y_0 = self.create_input('y_0', 2.0)
            x_0 = self.create_input('x_0', 2.0)

            # Create parameter for parameters a,b,g,d
            a = np.zeros((num_times, ))
            b = np.zeros((num_times, ))
            g = np.zeros((num_times, ))
            d = 0.5  # static parameter
            for t in range(num_times):
                a[t] = 1.0 + t/num_times/5.0
                b[t] = 0.5 + t/num_times/5.0
                g[t] = 2.0 + t/num_times/5.0 

            # Add to csdl model which are fed into ODE Model
            ai = self.create_input('a', a)
            bi = self.create_input('b', b)
            gi = self.create_input('g', g)
            di = self.create_input('d', d)

            # Timestep vector
            h_vec = np.ones(num_times-1)*h_stepsize
            h = self.create_input('h', h_vec)
            ode_system = ODESystemModel  # CSDL

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

            # ODE
            ode_problem.add_profile_output('intermediate_variable_1')
            ode_problem.add_profile_output('intermediate_variable_2')
            ode_problem.add_parameter('a', dynamic=True, shape=(num_times))
            ode_problem.add_parameter('b', dynamic=True, shape=(num_times))
            ode_problem.add_parameter('g', dynamic=True, shape=(num_times))
            ode_problem.add_parameter('d')
            ode_problem.add_state('y', 'dy_dt', initial_condition_name='y_0')
            ode_problem.add_state('x', 'dx_dt', initial_condition_name='x_0')
            ode_problem.add_times(step_vector='h')
            ode_problem.set_ode_system(ode_system, use_as_profile_output_system=True)
            self.add(ode_problem.create_solver_model(), 'subgroup')
            
            # Intermediate variables:
            po1 = self.declare_variable('intermediate_variable_1', shape=(num_times, 1)) # intermediate variables now exposed to the outer model
            po2 = self.declare_variable('intermediate_variable_2', shape=(num_times, 1)) # intermediate variables now exposed to the outer model
            self.register_output('exposed_iv1', po1*1.0)
            self.register_output('exposed_iv2', po2*1.0)

            self.register_output('out', csdl.pnorm(po1+po2))

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

    vals = {
        'out': sim['out'],
    }
    print(sim['out'])
    # exit()
    derivative_checks = sim.check_totals(of=['out'], wrt=['a', 'b', 'g','d','y_0','x_0'])

    sim.assert_check_partials(derivative_checks)
    derivative_checks = sim.compute_totals(of=['out'], wrt=['a', 'b', 'g','d','y_0','x_0'])
    for key in derivative_checks:
        print('derivative norm:', key, np.linalg.norm(derivative_checks[key]))
    # exit()
    # sim.check_totals(of =['y_out'], wrt = ['y_0', 'h'],compact_print=1)
    # exit()
    # sim.check_totals(of=['stage__y', 'state__dy_dt'], wrt='y_0')

    # exit()
    return_dict = {'output': vals, 'derivative_checks': derivative_checks}

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
