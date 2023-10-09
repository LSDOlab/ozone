from ozone.tests.test_2_general.run_ODE import run_ode
import pytest
from ozone.tests.test_2_general.utils_test import check_derivs, check_output, get_settings_dict

# ================================= NS non-sparse =================================


# def test_NS_timemarching():
#     settings_dictionary = get_settings_dict()
#     settings_dictionary['system'] = 'NSstd'
#     settings_dictionary['approach'] = 'time-marching checkpointing'
#     settings_dictionary['jvp_solver'] = 'iterative'
#     settings_dictionary['fwd_solver'] = 'iterative'
#     checks = run_ode(settings_dictionary)

#     check_output(checks['output'][0])
#     check_derivs(checks['derivative_checks'])


# def test_NS_timemarching_fwd_direct():
#     settings_dictionary = get_settings_dict()
#     settings_dictionary['jvp_solver'] = 'iterative'
#     settings_dictionary['approach'] = 'time-marching checkpointing'
#     settings_dictionary['fwd_solver'] = 'direct'
#     settings_dictionary['system'] = 'NSstd'

#     checks = run_ode(settings_dictionary)
#     check_output(checks['output'][0])
#     check_derivs(checks['derivative_checks'])


# def test_NS_timemarching_jvp_direct():
#     settings_dictionary = get_settings_dict()
#     settings_dictionary['jvp_solver'] = 'direct'
#     settings_dictionary['approach'] = 'time-marching checkpointing'
#     settings_dictionary['fwd_solver'] = 'iterative'
#     settings_dictionary['system'] = 'NSstd'

#     checks = run_ode(settings_dictionary)
#     check_output(checks['output'][0])

#     check_derivs(checks['derivative_checks'])


# def test_NS_timemarching_jvp_fwd_direct():
#     settings_dictionary = get_settings_dict()
#     settings_dictionary['approach'] = 'time-marching checkpointing'
#     settings_dictionary['jvp_solver'] = 'direct'
#     settings_dictionary['fwd_solver'] = 'direct'
#     settings_dictionary['system'] = 'NSstd'

#     checks = run_ode(settings_dictionary)
#     check_output(checks['output'][0])

#     check_derivs(checks['derivative_checks'])

# ================================= CSDL =================================


def test_CSDL_timemarching():
    settings_dictionary = get_settings_dict()
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['approach'] = 'time-marching checkpointing'
    settings_dictionary['jvp_solver'] = 'iterative'
    settings_dictionary['fwd_solver'] = 'iterative'
    checks = run_ode(settings_dictionary)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])


def test_CSDL_timemarching_fwd_direct():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = 'time-marching checkpointing'
    settings_dictionary['fwd_solver'] = 'direct'
    settings_dictionary['jvp_solver'] = 'iterative'
    settings_dictionary['system'] = 'CSDL'

    checks = run_ode(settings_dictionary)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])


def test_CSDL_timemarching_jvp_direct():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = 'time-marching checkpointing'
    settings_dictionary['jvp_solver'] = 'direct'
    settings_dictionary['fwd_solver'] = 'iterative'
    settings_dictionary['system'] = 'CSDL'

    checks = run_ode(settings_dictionary)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])


def test_CSDL_timemarching_jvp_fwd_direct():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = 'time-marching checkpointing'
    settings_dictionary['jvp_solver'] = 'direct'
    settings_dictionary['fwd_solver'] = 'direct'
    settings_dictionary['system'] = 'CSDL'

    checks = run_ode(settings_dictionary)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])


if __name__ == '__main__':
    print('--------------------------------------------------IMPLICIT--------------------------------------------------')
    # test_NS_timemarching()
    # total:  [1.04318422]
    # derivative norm: ('total', 'a') 0.0037963794392402257
    # derivative norm: ('total', 'x_0') 1.0181786974206675
    # derivative norm: ('total', 'h') 19.463407640235026
    test_CSDL_timemarching()
    # total:  [1.04318422]
    # derivative norm: ('total', 'a') 0.0037963794392402257
    # derivative norm: ('total', 'x_0') 1.0181786974206675
    # derivative norm: ('total', 'h') 19.463407640235026

    # test_NS_timemarching_fwd_direct()
    # test_NS_timemarching_jvp_direct()
