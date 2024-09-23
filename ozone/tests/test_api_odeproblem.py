import pytest
import numpy as np

def test_ode_problem_args():
    import ozone as ozone
    import csdl_alpha as csdl
    rec = csdl.Recorder()
    rec.start()

    with pytest.raises(TypeError):
        ozone.ODEProblem(ozone.methods.RK4(), 1)

    with pytest.raises(TypeError):
        ozone.ODEProblem(1, 'picard_iteration')

    # invalid sets/adds:
    picard_iteration = ozone.approaches.PicardIteration()
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), picard_iteration)

    # test add_state
    with pytest.raises(TypeError):
        x0 = csdl.Variable(name = 'x0', value = 2.0)
        ode_problem.add_state(1, x0)
    with pytest.raises(TypeError):
        x0 = csdl.Variable(name = 'x0', value = 2.0)
        ode_problem.add_state('x', 'a')
    with pytest.raises(TypeError):
        x0 = csdl.Variable(name = 'x0', value = 2.0)
        ode_problem.add_state('x', x0, store_history = 1)
    with pytest.raises(ValueError):
        x0 = csdl.Variable(name = 'x0', value = 2.0)
        ode_problem.add_state('x', x0, store_history = True)
        ode_problem.add_state('x', x0*1.0)

    # test add_dynamic_parameter
    with pytest.raises(TypeError):
        a = csdl.Variable(name = 'a', value = 2.0)
        ode_problem.add_dynamic_parameter(1, a)
    with pytest.raises(TypeError):
        a = csdl.Variable(name = 'a', value = 2.0)
        ode_problem.add_dynamic_parameter('a', 'a')
    with pytest.raises(ValueError):
        a = csdl.Variable(name = 'a', value = 2.0)
        ode_problem.add_dynamic_parameter('a', a)
        ode_problem.add_dynamic_parameter('a', a*1.0)
    
    # test set_timespan
    with pytest.raises(TypeError):
        ode_problem.set_timespan('a')
    with pytest.raises(ValueError):
        ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=1.0))
        ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=1.0))

    # invalid sets/adds combination
    # dynamic parameter wrong dimensions
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), picard_iteration)
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a', np.array([1.0, 2.0]))
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=[0.0, 1.0]))
    ode_problem.set_function(lambda x: None)
    with pytest.raises(ValueError):
        ode_problem.solve()

    # no ODE function
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), picard_iteration)
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a', np.array([1.0, 2.0]))
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=[0.0, 1.0]))
    with pytest.raises(ValueError):
        ode_problem.solve()

    # no timespan
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), picard_iteration)
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a', np.array([1.0, 2.0]))
    ode_problem.set_function(lambda x: None)
    with pytest.raises(ValueError):
        ode_problem.solve()

    # no states
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), picard_iteration)
    ode_problem.add_dynamic_parameter('a', np.array([1.0, 2.0]))
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=[0.0, 1.0]))
    ode_problem.set_function(lambda x: None)
    with pytest.raises(ValueError):
        ode_problem.solve()

    # solve twice
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), picard_iteration)
    ode_problem.add_dynamic_parameter('a', np.array([1.0, 2.0, 3.0]))
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=[0.0, 1.0]))
    
    def f(x):
        x.d_states['x'] = x.states['x']

    ode_problem.set_function(f)
    ode_problem.add_state('x', x0)
    ode_problem.solve()
    with pytest.raises(ValueError):
        ode_problem.solve()


def test_ode_function_api():
    import ozone as ozone
    import csdl_alpha as csdl
    rec = csdl.Recorder()
    rec.start()

    num_times = 10
    x0 = csdl.Variable(name = 'x0', value = np.ones((1,2)))
    a_dynamic = csdl.Variable(name = 'a_dynamic', value = np.ones((num_times,3)))
    a_static = csdl.Variable(name = 'a_static', value = np.ones((1,)))
    
    # solve once successfully
    def f(ozone_vars:ozone.ODEVars):
        a_contracted = csdl.sum(ozone_vars.dynamic_parameters['a_dynamic'], axes=(1,))
        ozone_vars.d_states['x'] = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static
        ozone_vars.field_outputs['x'] = ozone_vars.states['x']
        ozone_vars.profile_outputs['x'] = ozone_vars.states['x']
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.PicardIteration())
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a_dynamic', a_dynamic)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=0.1*np.ones((num_times-1,))))
    ode_problem.set_function(f)
    outputs = ode_problem.solve()

    # extra key:
    def f(ozone_vars:ozone.ODEVars):
        a_contracted = csdl.sum(ozone_vars.dynamic_parameters['a_dynamic'], axes=(1,))
        ozone_vars.d_states['x'] = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static
        ozone_vars.d_states['x1'] = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static
        ozone_vars.field_outputs['x'] = ozone_vars.states['x']
        ozone_vars.profile_outputs['x'] = ozone_vars.states['x']
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.PicardIteration())
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a_dynamic', a_dynamic)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=0.1*np.ones((num_times-1,))))
    ode_problem.set_function(f)
    with pytest.raises(KeyError):
        outputs = ode_problem.solve()

    # key not found:
    def f(ozone_vars:ozone.ODEVars):
        a_contracted = csdl.sum(ozone_vars.dynamic_parameters['a_dynamic'], axes=(1,))
        ozone_vars.d_states['x1'] = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static
        ozone_vars.field_outputs['x'] = ozone_vars.states['x']
        ozone_vars.profile_outputs['x'] = ozone_vars.states['x']
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.PicardIteration())
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a_dynamic', a_dynamic)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=0.1*np.ones((num_times-1,))))
    ode_problem.set_function(f)
    with pytest.raises(KeyError):
        outputs = ode_problem.solve()

    # state shape wrong:
    def f(ozone_vars:ozone.ODEVars):
        a_contracted = csdl.sum(ozone_vars.dynamic_parameters['a_dynamic'], axes=(1,))
        dx = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static
        ozone_vars.d_states['x'] = dx[0]
        ozone_vars.field_outputs['x'] = ozone_vars.states['x']
        ozone_vars.profile_outputs['x'] = ozone_vars.states['x']
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.PicardIteration())
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a_dynamic', a_dynamic)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=0.1*np.ones((num_times-1,))))
    ode_problem.set_function(f)
    with pytest.raises(ValueError):
        outputs = ode_problem.solve()

    # profile outputs shape wrong:
    def f(ozone_vars:ozone.ODEVars):
        a_contracted = csdl.sum(ozone_vars.dynamic_parameters['a_dynamic'], axes=(1,))
        dx = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static
        ozone_vars.d_states['x'] = dx
        ozone_vars.field_outputs['x'] = ozone_vars.states['x']
        ozone_vars.profile_outputs['x'] = ozone_vars.states['x'][0,:,:]
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.PicardIteration())
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a_dynamic', a_dynamic)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=0.1*np.ones((num_times-1,))))
    ode_problem.set_function(f)
    with pytest.raises(ValueError):
        outputs = ode_problem.solve()

    # field outputs shape wrong:
    def f(ozone_vars:ozone.ODEVars):
        a_contracted = csdl.sum(ozone_vars.dynamic_parameters['a_dynamic'], axes=(1,))
        dx = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static
        ozone_vars.d_states['x'] = dx
        ozone_vars.field_outputs['x'] = ozone_vars.states['x'][0,:,:]
        ozone_vars.profile_outputs['x'] = ozone_vars.states['x']
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.PicardIteration())
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a_dynamic', a_dynamic)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=0.1*np.ones((num_times-1,))))
    ode_problem.set_function(f)
    with pytest.raises(ValueError):
        outputs = ode_problem.solve()

    # args wrong
    def f(ozone_vars:ozone.ODEVars, d):
        a_contracted = csdl.sum(ozone_vars.dynamic_parameters['a_dynamic'], axes=(1,))
        dx = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static*d
        ozone_vars.d_states['x'] = dx
        ozone_vars.field_outputs['x'] = ozone_vars.states['x'][0,:,:]
        ozone_vars.profile_outputs['x'] = ozone_vars.states['x']
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.PicardIteration())
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a_dynamic', a_dynamic)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=0.1*np.ones((num_times-1,))))
    ode_problem.set_function(f)
    with pytest.raises(TypeError):
        outputs = ode_problem.solve()

    # kwargs wrong
    def f(ozone_vars:ozone.ODEVars, d):
        a_contracted = csdl.sum(ozone_vars.dynamic_parameters['a_dynamic'], axes=(1,))
        dx = -ozone_vars.states['x']*a_contracted.expand(ozone_vars.states['x'].shape, action='i->ijk')*a_static
        ozone_vars.d_states['x'] = dx
    ode_problem = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.PicardIteration())
    ode_problem.add_state('x', x0)
    ode_problem.add_dynamic_parameter('a_dynamic', a_dynamic)
    ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=0.1*np.ones((num_times-1,))))
    ode_problem.set_function(f,d1=1)
    with pytest.raises(TypeError):
        outputs = ode_problem.solve()

if __name__ == '__main__':
    test_ode_problem_args()
    test_ode_function_api()