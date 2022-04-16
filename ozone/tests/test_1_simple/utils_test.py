
import numpy as np
import pytest
# ================================= Functions =================================


def check_derivs(dict):
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('y_out', 'y_0'):
            assert pytest.approx(derivative, rel=1e-4) == 26.047954954623957
        elif key == ('y_out', 'h'):
            assert pytest.approx(derivative, rel=1e-4) == 38.575000915365756


def check_output(output):
    assert pytest.approx(output, rel=1e-5) == 13.0239779


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
