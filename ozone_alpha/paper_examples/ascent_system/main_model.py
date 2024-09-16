import numpy as np
import csdl_alpha as csdl
import ozone_alpha as ozone
import numpy as np

def ode_func(ozone_vars:ozone.FuncVars):
    g0 = 1.625
    R0 = 1738100
    vex = 3049.87
    w2 = 1.
    T = 1.8335634
    T_m = T/ozone_vars.states['m']
    n = ozone_vars.num_nodes

    ozone_vars.d_states['rx'] = ozone_vars.states['Vx'].reshape(-1,1)
    ozone_vars.d_states['ry'] = ozone_vars.states['Vy'].reshape(-1,1)
    ozone_vars.d_states['rz'] = ozone_vars.states['Vz'].reshape(-1,1)
    ozone_vars.d_states['Vx'] = (-w2*ozone_vars.states['rx'] + (T_m) * ozone_vars.dynamic_parameters['ux']).reshape(-1,1)
    ozone_vars.d_states['Vy'] = (-w2*ozone_vars.states['ry'] + (T_m) * ozone_vars.dynamic_parameters['uy']).reshape(-1,1)
    ozone_vars.d_states['Vz'] = (-w2*ozone_vars.states['rz'] + (T_m) * ozone_vars.dynamic_parameters['uz']).reshape(-1,1)
    ozone_vars.d_states['m'] = (-np.ones(n) * (T / vex) * np.sqrt(g0*R0)).reshape(-1,1)

def orbital_conditions(
        final_states,
        num:int,
        a:float, 
        ecc:float, 
        inc:float,
    ):

    rx_final = final_states.final_states['rx']
    ry_final = final_states.final_states['ry']
    rz_final = final_states.final_states['rz']
    Vx_final = final_states.final_states['Vx']
    Vy_final = final_states.final_states['Vy']
    Vz_final = final_states.final_states['Vz']
    m_final = final_states.final_states['m']

    r = csdl.concatenate((rx_final, ry_final, rz_final))
    V = csdl.concatenate((Vx_final, Vy_final, Vz_final))

    rxV1 = csdl.cross(r, V, axis=0)
    rxV2 = csdl.cross(r, V, axis=0)
    h = np.sqrt(a * (1 - ecc**2))

    c1 = csdl.inner(rxV1, rxV2) - a * (1 - ecc**2)
    c2 = csdl.norm(V, ord=2)**2/2. - 1. / csdl.norm(r, ord=2) + 1./(2*a)
    c3 = csdl.inner(np.array([0, 0, 1]), rxV1) - h * np.cos(inc)

    return c1, c2, c3

def build_recorder(approach:ozone.approaches._Approach, method:str, nt:int, plot:bool):

    # Start setup
    recorder = csdl.Recorder(inline = True, debug = True)
    recorder.start()

    # Initialize ODE problem:
    ode_problem = ozone.ODEProblem(method, approach)
 
    # Inputs:
    # - Timespan:
    tf = csdl.Variable(name='final_time', value=0.1) # OLD
    timespan = ozone.timespan.StepVector(start = 0.0, step_vector= (tf/(nt-1)).expand((nt-1,)))

    # - Control Inputs:
    ux = csdl.Variable(name='ux', value=-0.5, shape=(nt,))
    uy = csdl.Variable(name='uy', value=-0.5, shape=(nt,))
    uz = csdl.Variable(name='uz', value=0.5, shape=(nt,))

    # - State initial conditions:
    m_scale = 15103.
    rx_0 = csdl.Variable(name='rx_0', value = 0.061663494666509)
    ry_0 = csdl.Variable(name='ry_0', value = 0.055249439788994)
    rz_0 = csdl.Variable(name='rz_0', value = -1.004110522066614)
    Vx_0 = csdl.Variable(name='Vx_0', value = -0.465976472038572)
    Vy_0 = csdl.Variable(name='Vy_0', value = -0.416909510597800)
    Vz_0 = csdl.Variable(name='Vz_0', value = -0.084973690970470)
    m_0 = csdl.Variable(name='m_0', value = 11846.599614336959/m_scale)

    # Build ODE problem
    ode_problem.add_state('rx', initial_condition=rx_0, store_final=True, store_history=plot)
    ode_problem.add_state('ry', initial_condition=ry_0, store_final=True, store_history=plot)
    ode_problem.add_state('rz', initial_condition=rz_0, store_final=True, store_history=plot)
    ode_problem.add_state('Vx', initial_condition=Vx_0, store_final=True, store_history=plot)
    ode_problem.add_state('Vy', initial_condition=Vy_0, store_final=True, store_history=plot)
    ode_problem.add_state('Vz', initial_condition=Vz_0, store_final=True, store_history=plot)
    ode_problem.add_state('m', initial_condition=m_0, store_final=True, store_history=plot)
    ode_problem.add_dynamic_parameter('ux', ux)
    ode_problem.add_dynamic_parameter('uy', uy)
    ode_problem.add_dynamic_parameter('uz', uz)
    ode_problem.set_timespan(timespan=timespan)
    ode_problem.set_function(ode_func)

    # Integrate problem
    integrated_outputs = ode_problem.solve()

    # Post-process integration outputs:
    c1, c2, c3 = orbital_conditions(
        final_states = integrated_outputs,
        num = nt,
        a=1.04603, 
        ecc=0.0385, 
        inc=np.pi/2,
    )
    cu = ux**2 + uy**2 + uz**2 - 1.
    m_final = integrated_outputs.final_states['m']


    # Set up optimization variables
    # - Design variables
    ux.set_as_design_variable(lower=-1., upper=1.)
    uy.set_as_design_variable(lower=-1., upper=1.)
    uz.set_as_design_variable(lower=-1., upper=1.)
    tf.set_as_design_variable(lower=1.e0/1034.2)
    
    # - Objective
    tf.set_as_objective()

    # - Constraints
    c1.set_as_constraint(equals=0.)
    c2.set_as_constraint(equals=0.)
    c3.set_as_constraint(equals=0.)
    cu.set_as_constraint(equals=0.)
    m_final.set_as_constraint(lower=1./m_scale)

    # Clean up
    recorder.stop()

    if plot:
        integrated_outputs.states['rx'].add_name('full_rx')
        integrated_outputs.states['ry'].add_name('full_ry')
        integrated_outputs.states['rz'].add_name('full_rz')
        integrated_outputs.states['Vx'].add_name('full_Vx')
        integrated_outputs.states['Vy'].add_name('full_Vy')
        integrated_outputs.states['Vz'].add_name('full_Vz')
        tf.add_name('final_time')

    return recorder, ode_problem, nt