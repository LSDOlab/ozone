import numpy as np
import csdl_alpha as csdl
import ozone_alpha as ozone
import numpy as np
from ozone_alpha.paper_examples.trajectory_optimization.ode_func import ode_function
from ozone_alpha.paper_examples.trajectory_optimization.tonal import tonal

options = {}  # aircraft and mission parameter dictionary
# aircraft data
options['mass'] = 2000  # 3724 (kg)
options['wing_area'] = 19.6  # (m^2)
options['aspect_ratio'] = 12.13
options['wing_set_angle'] = 2  # (deg)
options['max_cruise_power'] = 468300  # (w)
options['max_lift_power'] = 103652  # (w)
options['span_efficiency'] = 0.8  # finite wing correction
options['cd_0'] = 0.02  # zero-lift drag coefficient
options['cruise_rotor_diameter'] = 2.1  # 2.7(m)
options['lift_rotor_diameter'] = 2.0  # 3 (m)
options['num_lift_rotors'] = 8
options['num_cruise_blades'] = 4  # 6
options['num_lift_blades'] = 2
options['energy_scale'] = 0.0001  # scale energy for plotting
# mission parameters
options['gravity'] = 9.81  # (m/s^2)
options['v_0'] = 5  # (m/s)
options['gamma_0'] = 0  # (rad)
options['h_0'] = 0  # (m)
options['x_0'] = 0  # (m)
options['alpha_0'] = 0  # (rad)
options['h_f'] = 300  # (m)
options['v_f'] = 43  # (m/s)
options['gamma_f'] = 0  # (rad)
options['max_spl'] = 105  # (db)

def build_recorder(num:int, approach:ozone.approaches._Approach, method:str):
    # Preprocessing:
    options['control_x_i'] = np.ones(num)*1800  # (rpm)
    options['control_z_i'] = np.linspace(1100, 0, num)  # np.ones(20)*800 # (rpm)
    options['control_alpha_i'] = np.linspace(0.7, 0.1, num)  # np.ones(20)*0 # (deg)
    options['dt'] = 2.5
    options['num'] = num

    # Start the model building process
    recorder = csdl.Recorder(inline = False, debug=True)
    recorder.start()

    # ODE settings:
    dt = csdl.Variable(name = 'dt', value = options['dt'])
    h_vec = dt.expand((num-1,))
    h_vec.add_name('step_vector')

    # Inputs:
    control_x = csdl.Variable(name='control_x', value = options['control_x_i'])
    control_z = csdl.Variable(name='control_z', value = options['control_z_i'])
    control_alpha = csdl.Variable(name='control_alpha', value = options['control_alpha_i'])
    # initial conditions for states
    v_0 = csdl.Variable(name='v_0', value = options['v_0'])
    gamma_0 = csdl.Variable(name='gamma_0', value = options['gamma_0'])
    h_0 = csdl.Variable(name='h_0', value = options['h_0'])
    x_0 = csdl.Variable(name='x_0', value = options['x_0'])
    e_0 = csdl.Variable(name='e_0', value=0)

    # Build actual model
    ode_problem = ozone.ODEProblem(method=method, approach=approach)
    # states:
    ode_problem.add_state('v', initial_condition=v_0, store_history = True, initial_guess=np.linspace(5, 45, num-1).reshape(-1,1))
    ode_problem.add_state('gamma', initial_condition=gamma_0, store_final = True,store_history = True, initial_guess=np.linspace(0, 0, num-1).reshape(-1,1))
    ode_problem.add_state('h', initial_condition=h_0, store_history = True, initial_guess=np.linspace(0, 300, num-1).reshape(-1,1))
    ode_problem.add_state('x', initial_condition=x_0, store_history = True, initial_guess=np.linspace(1.0, 1000, num-1).reshape(-1,1))
    ode_problem.add_state('e', initial_condition=e_0, store_history = True, initial_guess=np.linspace(1.0, 1000, num-1).reshape(-1,1))

    # dynamic paramters
    ode_problem.add_dynamic_parameter('control_alpha', control_alpha)
    ode_problem.add_dynamic_parameter('control_z', control_z)
    ode_problem.add_dynamic_parameter('control_x', control_x)

    # time span
    ode_problem.set_timespan(ozone.timespan.StepVector(start = 0.0, step_vector = h_vec))

    # ode_function
    ode_problem.set_function(ode_function, options = options)

    # Integrate
    integrated_outputs = ode_problem.solve()

    # #### Compute constraints, objectives and set design variables####:
    # Constraints:
    # 1) Final altitude constraint:
    final_height = integrated_outputs.states['h'][-1]
    final_height.set_as_constraint(lower=options['h_f'] - 0.1, upper=options['h_f'] + 0.1, scaler=0.1)
    # 2) Minimum altitude constraint:
    minimum_height = csdl.minimum(integrated_outputs.states['h'], rho = 20.0) # ::::::DIFFERENT FROM ORIGINAL::::::
    minimum_height.set_as_constraint(lower=options['h_0'] - 0.1, scaler=0.001)
    # 3) Final velocity constraint:
    final_velocity = integrated_outputs.states['v'][-1]
    final_velocity.set_as_constraint(lower=options['v_f'] - 0.001, upper=options['v_f'] + 0.001, scaler=0.1)
    # 4) Final flight path angle constraint:
    final_gamma = integrated_outputs.final_states['gamma'][-1]
    final_gamma.set_as_constraint(lower=options['gamma_f'] - 0.001, upper=options['gamma_f'] + 0.001, scaler=1.0)
    # 5) Acoustics constraint
    cruise_spl_150 = tonal(integrated_outputs, control_alpha, control_x, control_z, options)
    max_spl_150 = csdl.maximum(cruise_spl_150, rho = 20.0) # ::::::DIFFERENT FROM ORIGINAL::::::
    max_spl_150.set_as_constraint(upper=options['max_spl'], scaler=0.1)

    # Set design variables
    control_alpha.set_as_design_variable(lower=-1*np.pi/2, upper=np.pi/2, scaler=100)
    control_x.set_as_design_variable(lower=0, scaler=0.01)
    control_z.set_as_design_variable(lower=0, scaler=0.01)
    dt.set_as_design_variable(lower=0.2, scaler=1.0)
#  Objective                          : 1.2563134538239056
        # Num. of infeasible constraints     : 0
        # Sum. of infeasibilities            : 0.0
        # Num. of superbasic variables       : 109
        # Major iterations                   : 234
    # Set objective:
    energy = integrated_outputs.states['e'][-1]
    energy.set_as_objective(scaler=0.001)

    # For plotting:
    integrated_outputs.states['h'].add_name('h_plot')
    integrated_outputs.states['x'].add_name('x_plot')
    integrated_outputs.states['v'].add_name('v_plot')
    integrated_outputs.states['gamma'].add_name('gamma_plot')
    integrated_outputs.states['e'].add_name('e_plot')

    recorder.stop()
    return recorder, ode_problem, num