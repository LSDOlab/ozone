import csdl_alpha as csdl
import numpy as np

def tonal(
        integrated_outputs, 
        control_alpha,
        control_x,
        control_z,
        options,
    ):
    # Num time points
    num = options['num']
    
    # Inputs:
    cruise_ct = integrated_outputs.profile_outputs['cruisect']
    lift_ct = integrated_outputs.profile_outputs['liftct']
    z = integrated_outputs.states['h']
    alpha = control_alpha
    v = integrated_outputs.states['v']

    # declare rotor options from dictionary
    num_lift_rotors = options['num_lift_rotors']
    cruise_rotor_diameter = options['cruise_rotor_diameter']
    lift_rotor_diameter = options['lift_rotor_diameter']
    num_lift_blades = options['num_lift_blades']
    num_cruise_blades = options['num_cruise_blades']
    epsilon = 1

    # mean aerodynamic chord
    cruise_mac = 0.15 # (m)
    lift_mac = 0.15 # (m)

    # compute rotor area and disk area
    cruise_ab = cruise_mac*(cruise_rotor_diameter/2)*num_cruise_blades # rotor area
    lift_ab = lift_mac*(lift_rotor_diameter/2)*num_lift_blades # rotor area
    cruise_ad = np.pi*((cruise_rotor_diameter/2)**2)
    lift_ad = np.pi*((lift_rotor_diameter/2)**2)

    # compute blade solidity
    cruise_sigma = cruise_ab/cruise_ad
    lift_sigma = lift_ab/lift_ad

    # compute rotor speed
    omega_x = 2*np.pi*control_x/60 # (rad/s)
    omega_z = 2*np.pi*control_z/60 # (rad/s)
    cruise_rotor_speed = omega_x*cruise_rotor_diameter/2
    lift_rotor_speed = omega_z*lift_rotor_diameter/2

    # schlegel king and mull broadband noise model
    cval = (cruise_rotor_speed**6)*cruise_ab*((cruise_ct.flatten()/cruise_sigma)**2)
    cruise_spl_150 = 10*csdl.log(cval + epsilon, base = 10) - 36.7 #+ f_cruise[i]
    return cruise_spl_150