from ozone.tests.run_ode import run_ode
from ozone.tests.test_correctness.test_2_general.utils import (
    build_and_run_ode,
    check_derivs,
    check_output,
    get_settings_dict,
)
from ozone.approaches import PicardIteration, Collocation, TimeMarching, TimeMarchingCheckpoints
from ozone.methods import RK4, Trapezoidal

def test_CSDL_TimeMarching_explicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = TimeMarching()
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] =  RK4()
    checks = run_ode(settings_dictionary, build_and_run_ode)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])

def test_CSDL_TimeMarching_implicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = TimeMarching()
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] = Trapezoidal()
    checks = run_ode(settings_dictionary, build_and_run_ode)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])

def test_CSDL_TimeMarchingUniform_explicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = TimeMarchingCheckpoints(num_checkpoints=5)
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] =  RK4()
    checks = run_ode(settings_dictionary, build_and_run_ode)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])

def test_CSDL_TimeMarchingUniform_implicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = TimeMarchingCheckpoints(num_checkpoints=5)
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] = Trapezoidal()
    checks = run_ode(settings_dictionary, build_and_run_ode)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])

def test_CSDL_PicardIteration_explicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = PicardIteration()
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] =  RK4()
    checks = run_ode(settings_dictionary, build_and_run_ode)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])

def test_CSDL_PicardIteration_implicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = PicardIteration()
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] = Trapezoidal()
    checks = run_ode(settings_dictionary, build_and_run_ode)
    check_output(checks['output'])
    check_derivs(checks['derivative_checks'])

def test_CSDL_Collocation_explicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = Collocation()
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] =  RK4()
    checks = run_ode(settings_dictionary, build_and_run_ode)
    check_output(checks['output'])
    # check_derivs(checks['derivative_checks'])

def test_CSDL_Collocation_implicit():
    settings_dictionary = get_settings_dict()
    settings_dictionary['approach'] = Collocation()
    settings_dictionary['system'] = 'CSDL'
    settings_dictionary['num_method'] = Trapezoidal()
    checks = run_ode(settings_dictionary, build_and_run_ode)
    check_output(checks['output'])
    # check_derivs(checks['derivative_checks'])

if __name__ == '__main__':
    test_CSDL_TimeMarching_explicit()
    test_CSDL_TimeMarching_implicit()
    test_CSDL_TimeMarchingUniform_explicit()
    test_CSDL_TimeMarchingUniform_implicit()
    test_CSDL_PicardIteration_explicit()
    test_CSDL_PicardIteration_implicit()
    test_CSDL_Collocation_explicit()
    test_CSDL_Collocation_implicit()