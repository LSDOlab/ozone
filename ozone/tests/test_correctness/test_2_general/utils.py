
import numpy as np
import pytest
import csdl_alpha as csdl
import ozone as ozone

def build_and_run_ode(
        numerical_method:str,
        approach,
        nt:int,
    )->tuple[dict[str, csdl.Variable]]:
    num_times = nt
    h_stepsize = 0.001

    # Initial condition for state
    y_0 = csdl.Variable(name = 'y_0', value = 2.0)
    x_0 = csdl.Variable(name = 'x_0', value = 2.0)
    z_0 = csdl.Variable(name = 'z_0', value = np.array([[2.0, 1.0], [-1.0, -3.0]]))

    # Parameters:
    # Create parameter for parameters a,b,g,d
    a = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    b = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    g = np.zeros((num_times, ))  # dynamic parameter defined at every timestep
    e = np.zeros((num_times, 2, 2))  # dynamic parameter defined at every timestep
    d = 0.5  # static parameter
    for t in range(num_times):
        a[t] = 1.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
        b[t] = 0.5 + t/num_times/5.0  # dynamic parameter defined at every timestep
        g[t] = 2.0 + t/num_times/5.0  # dynamic parameter defined at every timestep
        e[t, :, :] = np.array([[0.3, 0.2], [0.1, -2.6]]) + t/num_times/5.0  # dynamic parameter defined at every timestep
    ai = csdl.Variable(name = 'a', value = a)
    bi = csdl.Variable(name = 'b', value = b)
    gi = csdl.Variable(name = 'g', value = g)
    ei = csdl.Variable(name = 'e', value = e)
    di = csdl.Variable(name = 'd', value = d)
    coeffs = csdl.Variable(name = 'coefficients', value = np.ones(num_times)/(num_times))

    # Step vector:
    h_vec = np.ones(num_times-1)*h_stepsize
    h_vec = csdl.Variable(name = 'h', value = h_vec)
    
    # ODE function:
    def f(ozone_vars:ozone.FuncVars, d_arg:csdl.Variable):
        a = ozone_vars.dynamic_parameters['a']
        b = ozone_vars.dynamic_parameters['b']
        g = ozone_vars.dynamic_parameters['g']
        e = ozone_vars.dynamic_parameters['e']
        x = ozone_vars.states['x']
        y = ozone_vars.states['y']
        z = ozone_vars.states['z']
        n = ozone_vars.num_nodes

        z_adjust1 = csdl.expand(csdl.reshape(z[:,0,1], (n,)), (n, 2, 2), 'i->ijk')
        dy_dt = a*y - b*y*x
        dx_dt = g*x*y-d_arg*x
        dz_dt = csdl.Variable(name = 'dz_dt', shape = (n, 2, 2), value = np.zeros((n, 2, 2)))
        for i in csdl.frange(n):
        # for i in range(n):
            temp_y = y[i]**2
            temp_a = a[i]
            temp_x = x[i]
            val = -z[i, :, :]/3.0*z_adjust1[i, :, :]+temp_y/2.0 + temp_a*e[i, :, :] + temp_x/2.0
            dz_dt = dz_dt.set(csdl.slice[i, :, :], value = val)

        ozone_vars.d_states['y'] = dy_dt
        ozone_vars.d_states['x'] = dx_dt
        ozone_vars.d_states['z'] = dz_dt

        # Profile/field outputs:
        z_adjust1 = csdl.reshape(z[:,0,1], (n,))
        z_adjust1 = csdl.expand(z_adjust1, (n, 2, 2), 'i->ijk')
        z_adjust2 = csdl.reshape(z[:,1,0], (n,))
        z_adjust2 = csdl.expand(z_adjust2, (n, 2, 2), 'i->ijk')

        profile_output_z = csdl.reshape(z[:, 1, 1], (n,))
        profile_output_x = (y/4.0)*y
        profile_output_x = x + profile_output_x
        profile_output_y = z*z_adjust1*z_adjust2 + e**2 + csdl.expand(a.flatten(), (n, 2, 2), 'i->ijk') + csdl.expand(d, (n, 2, 2))

        ozone_vars.profile_outputs['z'] = profile_output_z
        ozone_vars.profile_outputs['x'] = profile_output_x
        ozone_vars.profile_outputs['y'] = profile_output_y

        # field outputs:
        coeffs = ozone_vars.dynamic_parameters['coefficients']
        ozone_vars.field_outputs['y'] = coeffs*y
        ozone_vars.field_outputs['z'] = coeffs[:,0].expand(z.shape, action = 'i->ijk')*z

    ode_problem = ozone.ODEProblem(numerical_method, approach)
    ode_problem.add_state('y', initial_condition=y_0)
    ode_problem.add_state('x', initial_condition=x_0)
    ode_problem.add_state('z', initial_condition=z_0, store_history = True)
    ode_problem.add_dynamic_parameter('a', ai)
    ode_problem.add_dynamic_parameter('b', bi)
    ode_problem.add_dynamic_parameter('g', gi)
    ode_problem.add_dynamic_parameter('e', ei)
    ode_problem.add_dynamic_parameter('coefficients', coeffs)
    ode_problem.set_timespan(ozone.timespans.StepVector(start = 0.0, step_vector=h_vec))
    ode_problem.set_function(f, di)
    outputs = ode_problem.solve()

    # process outputs
    foy = outputs.field_outputs['y']
    pox = outputs.profile_outputs['x']
    poz = outputs.profile_outputs['z']
    poy = outputs.profile_outputs['y']
    z_int = outputs.states['z']
    temp = csdl.reshape(z_int[-1, 0, 1], (1,))

    outputs = {
        'total': pox[-1]+poz[-1]+foy[0]+temp/2.0, 
        'total2': csdl.norm(poy[-1, :, :] + poy[0, :, :])
    }

    inputs = {
        'a': ai,
        'x_0': x_0,
        'h':h_vec,
        'z_0': z_0,
        'e': ei
    }
    return inputs, outputs

# ================================= Functions =================================

def check_derivs(dict):
    num_checked = 0
    for key in dict:
        derivative = np.linalg.norm(dict[key])
        if key == ('total', 'a'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-3) == 0.00574943937426972
        elif key == ('total', 'x_0'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 1.08155650643438
        elif key == ('total', 'h'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 57.652844215047104 
        elif key == ('total2', 'a'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-3) == 1.754553251836068
        elif key == ('total2', 'x_0'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 0.027324051014
        elif key == ('total2', 'h'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 33.7649223785
        elif key == ('total2', 'z_0'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 9.09442575264
        elif key == ('total2', 'e'):
            num_checked += 1
            assert pytest.approx(derivative, rel=1e-4) == 6.880652370272789

    assert num_checked == 8
def check_output(output):
    # assert pytest.approx(output, rel=1e-4) == 2.80508725
    # assert pytest.approx(output, rel=1e-4) == 2.80503345
    # print(output['total'], type(output['total']))
    assert pytest.approx(output['total'], rel=1e-4) == 2.80611596
    assert pytest.approx(output['total2'], rel=1e-4) == 22.27526371713729


def get_settings_dict():

    settings_dictionary = {
        'approach': 'time-marching',
        'system': 'CSDL',
        'fwd_solver': 'iterative',
        'jvp_solver': 'iterative',
        'num_method': ozone.methods.Trapezoidal(),
        'benchmark': False,
        'store_jacs': False,
        'numtimes': 31  # DO NOT CHANGE
    }

    return settings_dictionary