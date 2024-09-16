import csdl_alpha as csdl
import ozone as ozone
import numpy as np

recorder = csdl.Recorder(inline = True, debug = False)
recorder.start()

# solve ODE:
# dy_dt = a(t)*y - b(t)*y*x
# dx_dt = g(t)*x*y - d*x
# a(t), b(t), g(t) are dynamic parameters,
# d is a static parameter, there isn't a need to define it as a dynamic parameter
# They need to be defined for every timestep

num_times = 10
x_0 = csdl.Variable(name = 'x_0', value = 2.0)
y_0 = csdl.Variable(name = 'y_0', value = 2.0)

# Create parameter for parameters a,b,g,d
a = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
b = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
g = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
d = 0.5  # static parameter
for t in range(num_times):
    a[t] = 1.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
    b[t] = 0.5 + t/num_times/5.0  # dynamic parameter defined at every timestep
    g[t] = 2.0 + t/num_times/5.0  # dynamic parameter defined at every timestep

# Add to csdl model which are fed into ODE Model
ai = csdl.Variable(name = 'a', value = a)
bi = csdl.Variable(name = 'b', value = b)
gi = csdl.Variable(name = 'g', value = g)
di = csdl.Variable(name = 'd', value = d)

# Timestep vector
h_stepsize = 0.1
h_vec = np.ones(num_times-1)*h_stepsize
h = csdl.Variable(name = 'h', value = h_vec)

# solve ODE:
# dy_dt = a(t)*y - b(t)*y*x
# dx_dt = g(t)*x*y - d*x
def ode_function(
        ozone_vars:ozone.FuncVars,
    ):
    a = ozone_vars.dynamic_parameters['a'] # a(t)
    b = ozone_vars.dynamic_parameters['b'] # b(t)
    g = ozone_vars.dynamic_parameters['g'] # g(t)
    x = ozone_vars.states['x'] # x
    y = ozone_vars.states['y'] # y
    d = di # di is not a function of time, so it doesn't need to be passed as a dynamic parameter
    dx_dt = (g*x*y - d*x)
    dy_dt = (a*y - b*y*x)

    ozone_vars.d_states['y'] = dy_dt # dy_dt
    ozone_vars.d_states['x'] = dx_dt # dx_dt

    ozone_vars.field_outputs['x'] = x # any outputs you want to record summed across time
    ozone_vars.profile_outputs['x'] = x  # any outputs you want to record across time

# ======================== Build ODE ========================:
# Choose approach:
approach = ozone.approaches.TimeMarching()
# approach = ozone.approaches.Collocation()
# approach = ozone.approaches.PicardIteration()

# Initialize ODE problem:
ode_problem = ozone.ODEProblem('RK4', approach)
# ode_problem = ozone.ODEProblem('ImplicitMidpoint', approach)

# Set inputs/states to your ODE problem.
# These variables become inputs to the ode_function
ode_problem.add_state('y', y_0, store_history=True, store_final=True)
ode_problem.add_state('x', x_0)
ode_problem.add_dynamic_parameter('a', ai)
ode_problem.add_dynamic_parameter('b', bi)
ode_problem.add_dynamic_parameter('g', gi)

ode_problem.set_timespan(ozone.timespans.StepVector(start=0.0, step_vector=h))
ode_problem.set_function(ode_function)

# Solve ODE
outputs = ode_problem.solve()
# ======================== Build ODE ========================:

norm = csdl.norm(outputs.states['y']) + csdl.norm(outputs.field_outputs['x']) + csdl.norm(outputs.profile_outputs['x']) 
# csdl.derivative_utils.verify_derivatives(norm, [x_0, y_0, di, ai], step_size=1e-6)

opt = isinstance(approach, ozone.approaches.Collocation)
if opt:
    from modopt import CSDLAlphaProblem
    from modopt import SLSQP

    x = csdl.Variable(name = 'x_dummy', value = 2.0)
    x1 = (x/2.0)**2.0
    x1.set_as_objective()
    x.set_as_design_variable()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        derivatives_kwargs={'loop':False}
    )
    sim.run_forward()
    sim.compute_optimization_derivatives()
    # sim.check_optimization_derivatives()
    # Instantiate your problem using the csdl Simulator object and name your problem
    prob = CSDLAlphaProblem(problem_name='dparam',simulator=sim)

    options = dict(
        ftol=1e-9
    )
    import time
    optimizer = SLSQP(prob, solver_options = options)
    start = time.time()
    optimizer.solve()
    end = time.time()
    optimizer.print_results()
    print('Elapsed time: ', end-start)

    recorder.execute()

# Print outputs
print(outputs)
print(norm.value)
assert np.isclose(outputs.states['y'].value[-1], outputs.final_states['y'].value)
assert np.isclose(norm.value, np.array([92.86879583]))
print('assertions passed')