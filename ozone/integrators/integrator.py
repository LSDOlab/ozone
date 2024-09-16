from ozone.glms.GLMs import get_integration_method
from ozone.collections import States, DynamicParameters, FieldOutputs, ProfileOutputs, IntegratorOutputs
from ozone.func_wrapper import FuncWrapper
from ozone.approaches import _Approach
from ozone.glms.GLMs import GLMMethod
from ozone.func_wrapper import FuncWrapper

import csdl_alpha as csdl
import numpy as np
import time
import scipy.sparse as sp


class IntegratorBase(object):
    def __init__(
            self,
            method:GLMMethod,
            approach:_Approach,
            ode_func:FuncWrapper,
            states:States,
            dynamic_parameters:DynamicParameters,
            num_times:int,
            step_vector:csdl.Variable,
        ):
        self.method:GLMMethod = method
        self.approach:_Approach = approach
        self.ode_func:FuncWrapper = ode_func
        self.states:States = states
        self.dynamic_parameters:DynamicParameters = dynamic_parameters
        
        self.num_times:int = num_times
        self.num_steps:int = num_times - 1
        self.step_vector:csdl.Variable = step_vector

    def initialize_outputs(self,):
        self.field_outputs:FieldOutputs = self.ode_func.field_outputs
        self.profile_outputs:ProfileOutputs = self.ode_func.profile_outputs

    def finalize_outputs(self):
        integrator_outputs = IntegratorOutputs()
        # State outputs:
        integrator_outputs.states = {state_name:state.solved_state for state_name,state in self.states.states.items() if state.output}
        integrator_outputs.final_states = {state_name:state.final_state for state_name,state in self.states.states.items() if state.store_final}
        
        # Condensed outputs:
        integrator_outputs.profile_outputs = {name:po.solution for name,po in self.profile_outputs.profile_outputs.items()}
        integrator_outputs.field_outputs = {name:fo.solution for name,fo in self.field_outputs.field_outputs.items()}
        return integrator_outputs