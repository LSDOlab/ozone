import numpy as np
import csdl_alpha as csdl
from ozone.utils.general import check_if_string, variablize, check_if_bool, get_type_string, check_if_method
from ozone.collections import States, DynamicParameters, IntegratorOutputs
from ozone.timespan import TimeSpan
from ozone.approaches import _Approach
from ozone.glms.GLMs import GLMMethod, get_integration_method
from ozone.func_wrapper import FuncWrapper
from ozone.methods import _Method

from typing import Callable, Union

class ODEProblem(object):

    def __init__(
            self,
            method:_Method,
            approach:_Approach,
        ) -> None:

        # validate numerical method
        check_if_method(method, 'method')
        self.method:GLMMethod = get_integration_method(method.name)

        # validate solution approach
        if not isinstance(approach, _Approach):
            raise TypeError(f'expected type \'_Approach\' for argument \'approach\' but got {get_type_string(approach)}')
        self.approach:_Approach = approach

        self.states:States = States()
        self.d_params:DynamicParameters = DynamicParameters()
        self.solved:bool = False

        # time span
        self.time_vector:csdl.Variable = None
        self.step_vector:csdl.Variable = None
        self.num_times:int = None

        # function
        self.f:FuncWrapper = None        

    def add_state(
            self,
            name:str,
            initial_condition:csdl.Variable,
            store_history:bool = False,
            store_final:bool = False,
            initial_guess:np.ndarray = None,
            scaler:Union[float, int, np.ndarray] = None,
        ) -> None:
        
        check_if_string(name, 'name')
        initial_condition = variablize(initial_condition)
        check_if_bool(store_history, 'store_history')
        check_if_bool(store_final, 'store_final')
        if initial_guess is not None:
            if not isinstance(initial_guess, np.ndarray):
                raise TypeError(f'expected type \'np.ndarray\' for argument \'initial_guess\' but got {get_type_string(initial_guess)}')
        # TODO: add error checking for scaler

        self.states.add(
            name,
            initial_condition,
            store_history,
            store_final,
            initial_guess,
            scaler,
        )

    def add_dynamic_parameter(
            self,
            name:str,
            parameter:csdl.Variable,
        ) -> None:
        check_if_string(name, 'name')
        parameter = variablize(parameter)

        self.d_params.add(
            name,
            parameter,
        )

    def set_timespan(
            self,
            timespan:TimeSpan,
        ) -> None:
        if not isinstance(timespan, TimeSpan):
            raise TypeError(f'expected type \'TimeSpan\' for argument \'timespan\' but got {get_type_string(timespan)}')
        if self.num_times is not None:
            raise ValueError('timespan has already been set.')
        self.time_vector, self.step_vector, self.num_times = timespan.finalize()

    def set_function(
            self,
            f:Callable,
            *args,
            **kwargs,
        ) -> None:
        if not callable(f):
            raise TypeError(f'expected type \'callable\' for argument \'f\' but got {get_type_string(f)}')
        self.f:FuncWrapper = FuncWrapper(f, args, kwargs)

    def solve(self)->IntegratorOutputs:
        # perform checks:
        # - can only solve once
        if self.solved:
            raise ValueError('solve has already been called.')
        
        # - callable f must be defined
        if self.f is None:
            raise ValueError('The function \'f\' must be set using the \'set_function\' method before solving the ODE.')
    
        # time vector, step_vector, num_times must be defined
        if self.num_times is None:
            raise ValueError('The integration time span must be set using the \'set_timespan\' method before solving the ODE.')

        # inputs must be validated
        self.states.validate(self.num_times)
        self.d_params.validate(self.num_times)

        # solve the ODE
        self.solved = True

        # create integrator object
        self.integrator = self.approach.integrator_class(
            method = self.method,
            approach = self.approach,
            ode_func = self.f,
            states = self.states,
            dynamic_parameters = self.d_params,
            num_times = self.num_times,
            step_vector = self.step_vector,
        )
        self.integrator.integrate()
        outputs = self.integrator.finalize_outputs()
        return outputs