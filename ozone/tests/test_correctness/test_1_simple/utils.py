
import numpy as np
import pytest
import csdl_alpha as csdl
import ozone as ozone

def f(ozone_vars:ozone.ODEVars):
    ozone_vars.d_states['y'] = -ozone_vars.states['y']

def build_and_run_ode(
        numerical_method:str,
        approach,
        nt:int,
    )->tuple[dict[str, csdl.Variable]]:
    num_times = nt
    h_stepsize = 0.01

    # Initial condition for state
    y_0 = csdl.Variable(name = 'y_0', value = 0.5)
    h_vec = np.ones(num_times-1)*h_stepsize
    h_vec = csdl.Variable(name = 'h', value = h_vec)
    
    ode_problem = ozone.ODEProblem(numerical_method, approach)
    ode_problem.add_state('y', initial_condition=y_0, store_history=True)
    ode_problem.set_timespan(ozone.timespans.StepVector(start = 0.0, step_vector=h_vec))
    ode_problem.set_function(f)
    integrated_outputs = ode_problem.solve()

    outputs = {
        'y_out': csdl.sum(integrated_outputs.states['y'])
    }

    inputs = {
        'y_0': y_0,
        'h':h_vec,
    }
    return inputs, outputs

# ================================= Functions =================================

def check_derivs(dict):
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('y_out', 'y_0'):
            assert pytest.approx(derivative, rel=1e-4) == 26.047954954623957
        elif key == ('y_out', 'h'):
            assert pytest.approx(derivative, rel=1e-4) == 38.575000915365756
        else:
            raise KeyError('Invalid key')

def check_output(outputs):
    assert pytest.approx(outputs['y_out'], rel=1e-5) == 13.0239779

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
