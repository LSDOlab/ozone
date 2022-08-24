
import numpy as np
import pytest
# ================================= Functions =================================


def check_derivs(dict):
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('total', 'a'):
            # assert pytest.approx(derivative, rel=1e-3) == 0.0057375166097572835
            assert pytest.approx(derivative, rel=1e-3) == 0.005722938448208581
        elif key == ('total', 'x_0'):
            # assert pytest.approx(derivative, rel=1e-4) == 1.0813055871226822
            assert pytest.approx(derivative, rel=1e-4) == 1.0813055871226822
        elif key == ('total', 'h'):
            # assert pytest.approx(derivative, rel=1e-4) == 57.2699201278553
            assert pytest.approx(derivative, rel=1e-4) == 57.2619945203023
        elif key == ('total2', 'a'):
            # assert pytest.approx(derivative, rel=1e-3) == 0.0057375166097572835
            assert pytest.approx(derivative, rel=1e-3) == 0.8732165148498291
        elif key == ('total2', 'x_0'):
            # assert pytest.approx(derivative, rel=1e-4) == 1.0813055871226822
            assert pytest.approx(derivative, rel=1e-4) == 0.009068049892864668
        elif key == ('total2', 'h'):
            # assert pytest.approx(derivative, rel=1e-4) == 57.2699201278553
            assert pytest.approx(derivative, rel=1e-4) == 15.741577032150477
        elif key == ('total2', 'z_0'):
            # assert pytest.approx(derivative, rel=1e-4) == 57.2699201278553
            assert pytest.approx(derivative, rel=1e-4) == 1.9900498337491679
        elif key == ('total2', 'e'):
            # assert pytest.approx(derivative, rel=1e-4) == 57.2699201278553
            assert pytest.approx(derivative, rel=1e-4) == 1.4149978079315917


def check_output(output):
    assert pytest.approx(output, rel=1e-4) == 2.80508725


def get_settings_dict():

    settings_dictionary = {
        'approach': 'time-marching',
        'system': 'CSDL',
        'fwd_solver': 'iterative',
        'jvp_solver': 'iterative',
        'num_method': 'Trapezoidal',
        'benchmark': False,
        'numtimes': 31  # DO NOT CHANGE
    }

    return settings_dictionary