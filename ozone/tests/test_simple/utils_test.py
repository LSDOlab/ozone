
import numpy as np
import pytest
# ================================= Functions =================================


def check_derivs(dict):
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('y_out', 'y_0'):
            assert pytest.approx(derivative, rel=1e-4) == 26.788803026944883
        elif key == ('y_out', 'h'):
            assert pytest.approx(derivative, rel=1e-4) == 40.30919627008279


def check_output(output):
    assert pytest.approx(output, rel=1e-5) == 13.39440151


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
