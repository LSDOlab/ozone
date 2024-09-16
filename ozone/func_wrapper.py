from typing import Callable
from csdl_alpha import Variable
from ozone.collections import ProfileOutputs, FieldOutputs

class FuncVars(object):
    def __init__(
            self,
            states,
            dynamic_parameters,
            num_nodes,
            ) -> None:
        self.states:dict[str,Variable] = states
        self.dynamic_parameters:dict[str,Variable] = dynamic_parameters
        self.num_nodes:int = num_nodes

        # Check to make sure inputs are correct shape, ensure first dimension is num_nodes for states and dynamic_parameters
        for state_name, state_input in self.states.items():
            if state_input.shape[0] != num_nodes:
                raise ValueError(f'INTERNAL ERROR: Expected shape ({num_nodes}, ...) for state \'{state_name}\' but got {state_input.shape}')
        for dynamic_parameter_name, dynamic_parameter_input in self.dynamic_parameters.items():
            if dynamic_parameter_input.shape[0] != num_nodes:
                raise ValueError(f'INTERNAL ERROR: Expected shape ({num_nodes}, ...) for dynamic parameter \'{dynamic_parameter_name}\' but got {dynamic_parameter_input.shape}')

        self.d_states:dict[str,Variable] = {}
        self.profile_outputs:dict[str,Variable] = {}
        self.field_outputs:dict[str,Variable] = {}

    def post_process(self):
        for state_name, state_input in self.states.items():
            if state_name not in self.d_states:
                raise KeyError(f'No state derivative specified for state \'{state_name}\'.')
            if self.d_states[state_name].shape != state_input.shape:
                raise ValueError(f'Expected shape {state_input.shape} for state derivative \'{state_name}\' but got {self.d_states[state_name].shape}')
        for d_state_name in self.d_states:
            if d_state_name not in self.states:
                raise KeyError(f'User set state derivative \'{d_state_name}\' but state \'{d_state_name}\' does not exist. Given states: {list(self.states.keys())}')

        for output_name, profile_output in self.profile_outputs.items():
            if profile_output.shape[0] != self.num_nodes:
                raise ValueError(f'profile output {output_name} expects shape ({self.num_nodes}, ...) but got {profile_output.shape}')
            if len(profile_output.shape) < 2:
                expanded = profile_output.expand((self.num_nodes, 1), action='i->ij')
                self.profile_outputs[output_name] = expanded

        for output_name, field_output in self.field_outputs.items():
            if field_output.shape[0] != self.num_nodes:
                raise ValueError(f'field output {output_name} expects shape ({self.num_nodes}, ...) but got {field_output.shape}')
            if len(field_output.shape) < 2:
                expanded = field_output.expand((self.num_nodes, 1), action='i->ij')
                self.field_outputs[output_name] = expanded

    # change print so that it prints out all the keys and shapes of the states and dynamic_parameters here:
    def __str__(self):
        prints = [
            ('States:', self.states),
            ('Dynamic Parameters:', self.dynamic_parameters),
            ('F(states):', self.d_states),
            ('Profile Outputs:', self.profile_outputs),
            ('Field Outputs:', self.field_outputs),
        ]

        string = '\nODE Function inputs/outputs:\n'
        for name, vars in prints:
            string += f'{name}\n'
            for key, value in vars.items():
                string += f'\t{key}: {value.shape}\n'
        return string

class FuncWrapper(object):
    
    def __init__(
            self,
            f:Callable,
            args:tuple,
            kwargs:dict,
        ) -> None:
        # keep track of outputs and function
        self.f:Callable = f
        self.profile_outputs:ProfileOutputs = ProfileOutputs()
        self.field_outputs:FieldOutputs = FieldOutputs()
        self.ran_once:bool = False

        # Optional user arguments
        self.args:tuple = args
        self.kwargs:dict = kwargs

    def evaluate(
            self,
            states:dict,
            parameters:dict,
            num_nodes:int,
        )->FuncVars:
        func_vars:FuncVars = FuncVars(states, parameters, num_nodes)
        self.f(func_vars, *self.args, **self.kwargs)
        func_vars.post_process()

        if self.ran_once == False:
            for output_name, profile_output_var in func_vars.profile_outputs.items():
                self.profile_outputs.add(output_name, shape = profile_output_var.shape[1:])
            for output_name, field_output in func_vars.field_outputs.items():
                self.field_outputs.add(output_name, shape = field_output.shape[1:])

        self.ran_once = True
        return func_vars