import numpy as np
import csdl_alpha as csdl
from ozone_alpha.paper_examples.trajectory_optimization.pitt_peters_rotor_surrogate import sm_ct, sm_cp

class CustomRotor(csdl.CustomExplicitOperation):
    def evaluate(self, nn:int, v_axial:csdl.Variable, v_tan:csdl.Variable):
        self.nn = nn

        # Inputs:
        self.declare_input('vAxial', v_axial)
        self.declare_input('vTan', v_tan)

        # Outputs:
        ct = self.create_output('ct', v_axial.shape)
        cp = self.create_output('cp', v_axial.shape)
        return ct, cp

    def compute(self, inputs, outputs):
        n = self.nn

        # surrogate model interpolation
        # point = np.array([[inputs[name+'vAxial'], inputs[name+'vTan']]]).reshape(1,2)
        # ct = sm_ct.predict_values(point)
        # cp = sm_cp.predict_values(point)
        ct = np.zeros((n,1))
        cp = np.zeros((n,1))
        for i in range(n):
            point = np.array([[inputs['vAxial'][i], inputs['vTan'][i]]]).reshape(1, 2)
            ct[i,0] = sm_ct.predict_values(point)
            cp[i,0] = sm_cp.predict_values(point)

        # define outputs
        outputs['ct'] = 1*ct
        outputs['cp'] = 1*cp

    def compute_derivatives(self, inputs, outputs, derivatives):
        n = self.nn
        """
        # compute derivatives
        point = np.array([[inputs['vAxial'], inputs['vTan']]]).reshape(1,2)
        dct_dvaxial = sm_ct.predict_derivatives(point, 0)
        dct_dvtan = sm_ct.predict_derivatives(point, 0)
        dcp_dvaxial = sm_cp.predict_derivatives(point, 0)
        dcp_dvtan = sm_cp.predict_derivatives(point, 0)

        # assign derivatives
        derivatives['ct', 'vAxial'] = dct_dvaxial
        derivatives['ct', 'vTan'] = dct_dvtan
        derivatives['cp', 'vAxial'] = dcp_dvaxial
        derivatives['cp', 'vTan'] = dcp_dvtan
        """
        dct_dvaxial = np.zeros((n))
        dct_dvtan = np.zeros((n))
        dcp_dvaxial = np.zeros((n))
        dcp_dvtan = np.zeros((n))
        for i in range(n):
            point = np.array([[inputs['vAxial'][i], inputs['vTan'][i]]]).reshape(1, 2)
            dct_dvaxial[i] = sm_ct.predict_derivatives(point, 0)
            dct_dvtan[i] = sm_ct.predict_derivatives(point, 1)
            dcp_dvaxial[i] = sm_cp.predict_derivatives(point, 0)
            dcp_dvtan[i] = sm_cp.predict_derivatives(point, 1)

        derivatives['ct', 'vAxial'] = np.diag(dct_dvaxial)
        derivatives['ct', 'vTan'] = np.diag(dct_dvtan)
        derivatives['cp', 'vAxial'] = np.diag(dcp_dvaxial)
        derivatives['cp', 'vTan'] = np.diag(dcp_dvtan)

def rotor(
        ozone_vars,
        v_axial:csdl.Variable,
        v_tan:csdl.Variable,
        rps:csdl.Variable,
        density:csdl.Variable,
        name:str,
        options:dict,
    ):
    # Pre-processing:
    nn = ozone_vars.num_nodes

    # CT, CP surrogate
    custom_rotor = CustomRotor()
    ct, cp = custom_rotor.evaluate(nn, v_axial, v_tan)
    ozone_vars.profile_outputs[f'{name}ct'] = ct

    # compute thrust and power
    s = rps # revolutions per SECOND
    d = options[name+'_rotor_diameter']
    rho = density

    # compute thrust and power
    thrust = (ct*rho*(s**2)*(d**4)).reshape(-1,1)
    power = (cp*rho*(s**3)*(d**5)).reshape(-1,1)
    return thrust, power