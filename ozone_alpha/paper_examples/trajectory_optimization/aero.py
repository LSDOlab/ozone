import numpy as np
import csdl_alpha as csdl
from ozone_alpha.paper_examples.trajectory_optimization.aero_surrogate import sm_cl, sm_cd


class CustomAirfoil(csdl.CustomExplicitOperation):
    def evaluate(self, nn:int, alpha_w:csdl.Variable):
        self.nn = nn

        # Inputs:
        self.declare_input('alpha_w', alpha_w)

        # Outputs:
        cl = self.create_output('cl', alpha_w.shape)
        cd = self.create_output('cd', alpha_w.shape)
        return cl, cd

    def compute(self, inputs, outputs):
        n = self.nn

        # surrogate model
        cl = np.zeros((n,1))
        cd = np.zeros((n,1))
        for i in range(n):
            a = np.array([inputs['alpha_w'][i]])
            cl[i,0] = sm_cl.predict_values(a)
            cd[i,0] = sm_cd.predict_values(a)

        outputs['cl'] = 1*cl
        outputs['cd'] = 1*cd

    def compute_derivatives(self, inputs, outputs, derivatives):
        n = self.nn

        dcl_dalpha = np.zeros((n))
        dcd_dalpha = np.zeros((n))
        for i in range(n):
            a = np.array([inputs['alpha_w'][i]])
            dcl_dalpha[i] = sm_cl.predict_derivatives(a, 0)
            dcd_dalpha[i] = sm_cd.predict_derivatives(a, 0)

        derivatives['cl', 'alpha_w'] = np.diag(dcl_dalpha)
        derivatives['cd', 'alpha_w'] = np.diag(dcd_dalpha)


def atmosphere(nn:int, h:csdl.Variable):
    n = nn
    g = 9.806 # m/(s^2)        
    a = -6.5E-3 # K/m        
    Ts = 288.16 # deg K @ sea level        
    Ps = 1.01325E5 # Pascals at sea level        
    rhoS = 1.225 # kg/m^3 at sea level        
    R = 287 # J/(Kg-K) gas constant        
    temperature = Ts + a*h        
    pressure = Ps*((temperature/Ts)**((-g)/(a*R)))
    density = rhoS*((temperature/Ts)**(-((g/(a*R)) + 1)))
    return pressure.reshape(-1,1), density.reshape(-1,1)

def aero(
        ozone_vars,
        density:csdl.Variable,
        options:dict,
    ):
    nn = ozone_vars.num_nodes

    # pre-processing:
    alpha = ozone_vars.dynamic_parameters['control_alpha']
    alpha_w = np.deg2rad(options['wing_set_angle']) + alpha
    alpha_w = alpha_w.reshape(-1,1)

    # CL, CD surrogate
    airfoil = CustomAirfoil()
    cl, cd = airfoil.evaluate(nn, alpha_w)
    ozone_vars.profile_outputs['cl'] = cl
    ozone_vars.profile_outputs['cd'] = cd

    # lift, drag computation
    s = options['wing_area']
    cd_0 = options['cd_0']
    e = options['span_efficiency']
    aspect_ratio = options['aspect_ratio']
    velocity = ozone_vars.states['v']

    q = 0.5*density*(velocity**2)
    cd = (cd_0 + cd).reshape(-1,1)
    return q*s*cl, q*s*cd