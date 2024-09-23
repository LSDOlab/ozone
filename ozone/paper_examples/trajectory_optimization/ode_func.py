import numpy as np
import csdl_alpha as csdl
import ozone as ozone
import numpy as np

# Disciplines:
from ozone.paper_examples.trajectory_optimization.aero import aero, atmosphere
from ozone.paper_examples.trajectory_optimization.rotor import rotor

def ode_function(ozone_vars:ozone.ODEVars, options):
    # Pre-processing
    num_nodes = ozone_vars.num_nodes
    v = ozone_vars.states['v']
    gamma = ozone_vars.states['gamma']
    h = ozone_vars.states['h']
    x = ozone_vars.states['x']
    e = ozone_vars.states['e']

    # parameters are inputs
    control_x = ozone_vars.dynamic_parameters['control_x']
    control_z = ozone_vars.dynamic_parameters['control_z']
    alpha = ozone_vars.dynamic_parameters['control_alpha']

    m = options['mass']
    g = options['gravity']
    num_lift_rotors = options['num_lift_rotors']

    # Atmospheric variables:
    pressure, density = atmosphere(num_nodes, ozone_vars.states['h'])

    # Aero:
    L, D = aero(ozone_vars, density, options)

    # Rotors:
    # - Cruise:
    name = 'cruise'
    v_axial = v*csdl.cos(alpha)
    v_tan = v*csdl.sin(alpha)
    rps = 1*control_x/60 # rotations per second
    cruise_thrust, cruise_power = rotor(
        ozone_vars,
        v_axial,
        v_tan,
        rps,
        density,
        name,
        options,
    )
    # - Lift:
    name = 'lift'
    v_axial = v*csdl.sin(alpha)
    v_tan = v*csdl.cos(alpha)
    rps = 1*control_z/60 # rotations per second
    lift_thrust_single, lift_power_single = rotor(
        ozone_vars,
        v_axial,
        v_tan,
        rps,
        density,
        name,
        options,
    )
    lift_thrust = (lift_thrust_single*num_lift_rotors).reshape(-1,1)
    lift_power = (lift_power_single*num_lift_rotors).reshape(-1,1)
    
    # system of ODE's
    TC = cruise_thrust
    TL = lift_thrust

    dv = (TC/m)*csdl.cos(alpha) + (TL/m)*csdl.sin(alpha) - (D/m) - g*csdl.sin(gamma)
    dgamma = (TC/(m*v))*csdl.sin(alpha) + (TL/(m*v))*csdl.cos(alpha) + (L/(m*v)) - (g*csdl.cos(gamma)/v)
    dh = v*csdl.sin(gamma)
    dx = v*csdl.cos(gamma)
    de = options['energy_scale']*(cruise_power + lift_power)

    ozone_vars.d_states['v'] = dv.reshape(-1,1)
    ozone_vars.d_states['gamma'] = dgamma.reshape(-1,1)
    ozone_vars.d_states['h'] = dh.reshape(-1,1)
    ozone_vars.d_states['x'] = dx.reshape(-1,1)
    ozone_vars.d_states['e'] = de.reshape(-1,1)

    # dv.print_on_update(f'dv')
    # dgamma.print_on_update(f'dgamma')
    # dh.print_on_update(f'dh')
    # dx.print_on_update(f'dx')
    # de.print_on_update(f'de')

    ozone_vars.profile_outputs['lift'] = L
    ozone_vars.profile_outputs['drag'] = D
    ozone_vars.profile_outputs['cruisepower'] = cruise_power
    ozone_vars.profile_outputs['liftpower'] = lift_power
