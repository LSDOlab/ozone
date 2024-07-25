
import numpy as np
import pytest
# ================================= Functions =================================


def check_derivs(dict):
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('y_out', 'y_0'):
            assert pytest.approx(derivative, rel=1e-4) ==555.6297961976778
        elif key == ('y_out', 'h'):
            assert pytest.approx(derivative, rel=1e-4) == 12570.03729508261
        elif key == ('y_out', 'static_param'):
            assert pytest.approx(derivative, rel=1e-4) == 43.805547522344185
        elif key == ('y_out', 'dynamic_param'):
            assert pytest.approx(derivative, rel=1e-4) == 203.55862089571335

def check_output(output):
    assert pytest.approx(output, rel=1e-5) == 6794.12754072


def get_settings_dict():

    settings_dictionary = {
        'approach': 'time-marching',
        'system': 'CSDL',
        'fwd_solver': 'iterative',
        'jvp_solver': 'iterative',
        'num_method': 'Trapezoidal',
        'benchmark': False,
        'numtimes': 30  # DO NOT CHANGE
    }

    return settings_dictionary
