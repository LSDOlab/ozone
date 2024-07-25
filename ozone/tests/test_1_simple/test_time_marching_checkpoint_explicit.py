from ozone.tests.test_1_simple.run_ODE import run_ode
from ozone.tests.test_1_simple.utils_test import check_derivs, check_output, get_settings_dict

import pytest

# ================================= NS non-sparse =================================


# def test_NS_timemarching_explicit():
#     settings_dictionary = get_settings_dict()
#     settings_dictionary['approach'] = 'time-marching checkpointing'
#     settings_dictionary['system'] = 'NSstd'
#     settings_dictionary['num_method'] = 'RK4'
#     checks = run_ode(settings_dictionary)

#     check_output(checks['output'][0])
#     check_derivs(checks['derivative_checks'])


# ================================= CSDL =================================


def test_CSDL_timemarching_explicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = 'time-marching checkpointing'
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] = 'RK4'
    checks = run_ode(settings_dictionary)
    check_output(checks['output'][0])
    check_derivs(checks['derivative_checks'])


# ================================= Functions =================================


if __name__ == '__main__':
    print('--------------------------------------------------EXPLICIT--------------------------------------------------')
    test_NS_timemarching_explicit()
    # total:  [1.04318295]
    # derivative norm: ('total', 'a') 0.0038016174938430344
    # derivative norm: ('total', 'x_0') 1.0181788066850965
    # derivative norm: ('total', 'h') 19.462752682477294

    test_CSDL_timemarching_explicit()
    # total:  [1.04318295]
    # derivative norm: ('total', 'a') 0.0038016174938430344
    # derivative norm: ('total', 'x_0') 1.0181788066850965
    # derivative norm: ('total', 'h') 19.462752682477294
