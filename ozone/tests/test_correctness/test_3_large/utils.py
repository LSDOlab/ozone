
import numpy as np
import pytest
import csdl_alpha as csdl
import ozone as ozone

STATE_SIZE = 500
def build_and_run_ode(
        numerical_method:str,
        approach,
        nt:int,
    )->tuple[dict[str, csdl.Variable]]:
    num_times = nt
    h_stepsize = 0.01

    # Initial condition for state
    y_0 = csdl.Variable(name = 'y_0', value = 0.5*np.ones((STATE_SIZE,)))
    sp = csdl.Variable(name = 'static_param', value = np.linspace(0.1, 1.1, STATE_SIZE))
    dp = csdl.Variable(name = 'dynamic_param', value = np.linspace(0.5, 0.8, num_times).reshape((num_times,1)))

    # Step vector:
    h_vec = np.ones(num_times-1)*h_stepsize
    h_vec = csdl.Variable(name = 'h', value = h_vec)
    
    # ODE function:
    def f(ozone_vars:ozone.ODEVars):
        y = ozone_vars.states['y']
        n = ozone_vars.num_nodes
        dp = ozone_vars.dynamic_parameters['dynamic_param']
        expanded_sp = csdl.expand(sp, (n, STATE_SIZE,), 'i->ai')
        expanded_dp = csdl.expand(dp.flatten(), (n, STATE_SIZE,), 'i->ia')
        ozone_vars.d_states['y'] = -(y*(expanded_dp+expanded_sp))**2

    ode_problem = ozone.ODEProblem(numerical_method, approach)
    ode_problem.add_state('y', initial_condition=y_0, store_history=True)
    ode_problem.add_dynamic_parameter('dynamic_param', dp)
    ode_problem.set_timespan(ozone.timespans.StepVector(start = 0.0, step_vector=h_vec))
    ode_problem.set_function(f)
    outputs = ode_problem.solve()

    outputs = {
        'y_out': csdl.sum(outputs.states['y']), 
    }

    inputs = {
        'y_0': y_0,
        'h': h_vec,
        'static_param':sp,
        'dynamic_param': dp,
    }
    return inputs, outputs

# ================================= Functions =================================

def check_derivs(dict):
    num_checked = 0
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('y_out', 'y_0'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) ==555.6297961976778
        elif key == ('y_out', 'h'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 12570.03729508261
        elif key == ('y_out', 'static_param'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 43.805547522344185
        elif key == ('y_out', 'dynamic_param'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 203.55862089571335
    assert num_checked == 4

def check_output(outputs):
    assert pytest.approx(outputs['y_out'], rel=1e-5) == 6794.12754072


def get_settings_dict():

    settings_dictionary = {
        'approach': 'time-marching',
        'system': 'CSDL',
        'fwd_solver': 'iterative',
        'jvp_solver': 'iterative',
        'num_method': ozone.methods.Trapezoidal(),
        'benchmark': False,
        'numtimes': 30  # DO NOT CHANGE
    }

    return settings_dictionary