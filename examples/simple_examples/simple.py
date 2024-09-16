
import csdl_alpha as csdl
import ozone_alpha as ozone
import numpy as np

recorder = csdl.Recorder(inline = True, debug = False)
recorder.start()

# solve ODE:
# dy_dt = a(t)*y - b(t)*y*x
# dx_dt = g(t)*x*y - d*x

num_times = 10 # number of time points
# Create initial conditions
x_0 = csdl.Variable(name = 'x_0', value = 2.0)
y_0 = csdl.Variable(name = 'y_0', value = 2.0)

# Create dynamic parameter 'a', static parameter 'd'
# dynamic parameter must be defined for every timestep
a = np.zeros((num_times, )) 
for t in range(num_times):
    a[t] = 1.0 + t/num_times/5.0  
a_dynamic = csdl.Variable(name = 'a', value = a) 
# static parameter
d_static = csdl.Variable(name = 'd', value = 0.5) 

# Timestep vector with 9 timesteps of size 0.1
h = csdl.Variable(name = 'h', value = np.full(num_times-1, 0.1))

def ode_function(ozone_vars:ozone.FuncVars, d:csdl.Variable):
    a = ozone_vars.dynamic_parameters['a'] # a(t)
    x = ozone_vars.states['x'] # x
    y = ozone_vars.states['y'] # y

    ozone_vars.d_states['y'] = a*y - 0.5*y*x # dy_dt
    ozone_vars.d_states['x'] = 2.0*x*y - d*x # dx_dt

    # any outputs you want to record summed across time
    ozone_vars.field_outputs['field_x'] = x
    # any outputs you want to record across time
    ozone_vars.profile_outputs['x_plus_y'] = x + y

# Choose approach and method
approach = ozone.approaches.TimeMarching()
# approach = ozone.approaches.PicardIteration()
# approach = ozone.approaches.Collocation()

method = ozone.methods.RK4()
# method = ozone.methods.ImplicitMidpoint()

# Initialize ODE problem:
ode_problem = ozone.ODEProblem(method, approach)

# Set inputs/states to your ODE problem.
# Define states with their initial conditions
ode_problem.add_state('y', y_0)
ode_problem.add_state('x', x_0)
# Define dynamic parameters
ode_problem.add_dynamic_parameter('a', a_dynamic)
# Define time span
ode_problem.set_timespan(
    ozone.timespans.StepVector(start=0.0, step_vector=h))
# pass any arguments to ode_function
ode_problem.set_function(ode_function, d_static)

# Solve ODE
outputs = ode_problem.solve()

# Get the maximum of the sum of 'x' and 'y' across time using CSDL
x_plus_y = outputs.profile_outputs['x_plus_y']

# Return CSDL variable representing the derivatives of the maximum
# of the sum of 'x' and 'y' across time with respect to 
# dynamic parameters, static parameters, 
# time steps, and initial conditions
max_x_plus_y = csdl.maximum(x_plus_y)
dsumxy_da = csdl.derivative(max_x_plus_y, a_dynamic) 
dsumxy_dd = csdl.derivative(max_x_plus_y, d_static) 
dsumxy_dh = csdl.derivative(max_x_plus_y, h) 
dsumxy_dx0 = csdl.derivative(max_x_plus_y, x_0)

# Print the results
print('x + y:            ', x_plus_y.value.flatten())
print('Max of x + y:     ', max_x_plus_y.value.flatten())
print('d(max(x+y))/da    ', dsumxy_da.value.flatten())
print('d(max(x+y))/dd    ', dsumxy_dd.value.flatten())
print('d(max(x+y))/dh    ', dsumxy_dh.value.flatten())
print('d(max(x+y))/dx0   ', dsumxy_dx0.value.flatten())


