import numpy as np
import csdl_alpha as csdl
import ozone as ozone
import numpy as np

def ode_func(ozone_vars:ozone.ODEVars):
    x0 = ozone_vars.states['x0']
    x1 = ozone_vars.states['x1']
    j = ozone_vars.states['J']
    u = ozone_vars.dynamic_parameters['u']

    ozone_vars.d_states['x0'] = ((1.0 + (- 1.0) * (x1**2.0))*x0 + (-1*x1) + u).reshape(-1,1)
    ozone_vars.d_states['x1'] = (x0*1.0).reshape(-1,1)
    ozone_vars.d_states['J'] = (x0**2.0 + x1**2.0 + u**2.0).reshape(-1,1)

def build_recorder(
        approach:ozone.approaches._Approach,
        method:str, 
        nt:int, 
        tf:float,
        plot:bool,
    ):

    # Start setup
    recorder = csdl.Recorder(inline = False)
    recorder.start()

    # Build ODE:
    # Inputs:
    x0_0 = csdl.Variable(name='x0_0', value=1.0)
    x1_0 = csdl.Variable(name='x1_0', value=1.0)
    J_0 = csdl.Variable(name='J_0', value=0.0)
    u = csdl.Variable(name='u', value=np.ones((nt, 1))*(-0.75))
    h = csdl.Variable(name='h', value=np.ones((nt-1))*(tf/(nt-1)))

    # Build ODE problem
    ode_problem = ozone.ODEProblem(method, approach)
    ode_problem.add_state('x0', initial_condition = x0_0, store_history=plot, initial_guess=np.linspace(1.0,0.0,nt-1).reshape(-1,1))
    ode_problem.add_state('x1', initial_condition = x1_0, store_history=plot, initial_guess=np.linspace(1.0,0.0,nt-1).reshape(-1,1))
    ode_problem.add_state('J', initial_condition = J_0, store_history=plot, store_final=True, initial_guess=np.linspace(0.0,1.0,nt-1).reshape(-1,1))
    ode_problem.set_timespan(ozone.timespan.StepVector(start = 0.0, step_vector=h))
    ode_problem.add_dynamic_parameter('u', u)
    ode_problem.set_function(ode_func)

    # Integrate problem
    integrated_outputs = ode_problem.solve()

    # Process optimization variables:
    J_end = integrated_outputs.final_states['J']

    # Set optimization variables
    J_end.set_as_objective()
    u.set_as_design_variable(lower=-0.75, upper=1.0)

    # Clean up
    recorder.stop()

    if plot:
        integrated_outputs.states['x0'].add_name('full_x0')
        integrated_outputs.states['x1'].add_name('full_x1')
        integrated_outputs.states['J'].add_name('full_J')
        h.add_name('full_h')
        u.add_name('full_u')

    return recorder, ode_problem, nt