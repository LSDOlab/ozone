
import numpy as np
import pytest
# ================================= Functions =================================


def check_derivs(dict):
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('out', 'a'):
            assert pytest.approx(derivative, rel=1e-4) == 29.568642
        elif key == ('out', 'b'):
            assert pytest.approx(derivative, rel=1e-4) == 42.875
        elif key == ('out', 'd'):
            assert pytest.approx(derivative, rel=1e-3) == 0.083783
        elif key == ('out', 'g'):
            assert pytest.approx(derivative, rel=1e-2) == 0.058223531
        elif key == ('out', 'x_0'):
            assert pytest.approx(derivative, rel=1e-2) == 0.2013962
        elif key == ('out', 'y_0'):
            assert pytest.approx(derivative, rel=1e-4) == 31.45409

def check_output(output):
    assert pytest.approx(output['out'], rel=1e-3) == 88.5096


def get_settings_dict():

    settings_dictionary = {
        'approach': 'time-marching',
        'system': 'CSDL',
        'fwd_solver': 'iterative',
        'jvp_solver': 'iterative',
        'num_method': 'Trapezoidal',
        'benchmark': False,
        'numtimes': 10  # DO NOT CHANGE
    }

    return settings_dictionary
