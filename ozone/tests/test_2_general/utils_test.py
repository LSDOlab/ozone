
import numpy as np
import pytest
# ================================= Functions =================================


def check_derivs(dict):
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('total', 'a'):
            # assert pytest.approx(derivative, rel=1e-3) == 0.0057375166097572835
            # assert pytest.approx(derivative, rel=1e-3) == 0.005722938448208581
            assert pytest.approx(derivative, rel=1e-3) == 0.00574943937426972

        elif key == ('total', 'x_0'):
            # assert pytest.approx(derivative, rel=1e-4) == 1.0813055871226822
            # assert pytest.approx(derivative, rel=1e-4) == 1.0813055871226822
            assert pytest.approx(derivative, rel=1e-4) == 1.08155650643438

        elif key == ('total', 'h'):
            # assert pytest.approx(derivative, rel=1e-4) == 57.2699201278553
            # assert pytest.approx(derivative, rel=1e-4) == 57.2619945203023
            assert pytest.approx(derivative, rel=1e-4) == 57.652844215047104 
        elif key == ('total2', 'a'):
            # assert pytest.approx(derivative, rel=1e-3) == 0.0057375166097572835
            # assert pytest.approx(derivative, rel=1e-3) == 2.5184462112810233
            assert pytest.approx(derivative, rel=1e-3) == 1.754553251836068

        elif key == ('total2', 'x_0'):
            # assert pytest.approx(derivative, rel=1e-4) == 1.0813055871226822
            # assert pytest.approx(derivative, rel=1e-4) == 0.0262064210650258
            assert pytest.approx(derivative, rel=1e-4) == 0.027324051014

        elif key == ('total2', 'h'):
            # assert pytest.approx(derivative, rel=1e-4) == 57.2699201278553
            # assert pytest.approx(derivative, rel=1e-4) == 22.765760735534432
            assert pytest.approx(derivative, rel=1e-4) == 33.7649223785
        elif key == ('total2', 'z_0'):
            # assert pytest.approx(derivative, rel=1e-4) == 57.2699201278553
            # assert pytest.approx(derivative, rel=1e-4) == 1.9900498337491679
            assert pytest.approx(derivative, rel=1e-4) == 9.09442575264
        elif key == ('total2', 'e'):
            # assert pytest.approx(derivative, rel=1e-4) == 57.2699201278553
            # assert pytest.approx(derivative, rel=1e-4) == 5.1409600578737615
            assert pytest.approx(derivative, rel=1e-4) == 6.880652370272789



def check_output(output):
    # assert pytest.approx(output, rel=1e-4) == 2.80508725
    # assert pytest.approx(output, rel=1e-4) == 2.80503345
    print(output['total'], type(output['total']))
    assert pytest.approx(output['total'], rel=1e-4) == 2.80611596
    assert pytest.approx(output['total2'], rel=1e-4) == 22.27526371713729


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


# (of,wrt)              calc norm                relative error             absolute error
# ------------------------------------------------------------------------------------------------
# ('total', 'a')        0.00574943937426972      1.497920643760708e-06      8.612212811099136e-09
# ('total', 'e')        0.006666024571224257     8.222145523284875e-07      5.4809018587384755e-09
# ('total', 'h')        57.652844215047104       9.643947533307168e-07      5.5600153900106813e-05
# ('total', 'x_0')      1.08155650643438         2.595413067255738e-09      2.8070858970608015e-09
# ('total', 'z_0')      1.1174468456567688       4.14747416716541e-09       4.6345819167821394e-09
# ('total2', 'a')       1.754553251836068        4.397066392873222e-08      7.71488746553991e-08
# ('total2', 'e')       6.880652370272789        2.118630380550011e-07      1.4577556166090057e-06
# ('total2', 'h')       33.76492237850695        5.980662932118887e-06      0.00020193782617217455
# ('total2', 'x_0')     0.02732405101421452      1.7905687000977631e-07     4.892559926639173e-09
# ('total2', 'z_0')     9.09442575264479         1.3659160219208274e-07     1.2422221108537005e-06
# ------------------------------------------------------------------------------------------------