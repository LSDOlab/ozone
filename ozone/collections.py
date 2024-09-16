from ozone.utils.general import get_general_expand_string

import csdl_alpha as csdl
import numpy as np
from typing import Union

class State(object):
    def __init__(
            self,
            name:str,
            initial_condition:csdl.Variable,
            output:bool,
            store_final:bool,
            initial_guess:np.ndarray,
            scaler,
        ) -> None:
        self.name:str = name
        self.initial_condition:csdl.Variable = initial_condition
        self.output:bool = output
        self.store_final:bool = store_final
        self.initial_guess:np.ndarray = initial_guess
        self.scaler = scaler

        # shape information
        self.shape:tuple[int] = initial_condition.shape
        self.size:int = initial_condition.size
        self.solved_shape:tuple[int] = None

        # constant matrices
        self.constants:dict[str, Union[np.ndarray, 'sparse']] = {}

        # Outputs:
        self.solved_state:csdl.Variable = None
        self.final_state:csdl.Variable = None

        # for vector-based integrator:
        self.expanded_step_vector:csdl.Variable = None
        self.vectorized_initial_condition:csdl.Variable = None

        # for picard iteration:
        self.implicit_variable:csdl.Variable = None

        # for collocation:
        self.state_dv:csdl.Variable = None
        self.stage_dv:csdl.Variable = None
        self.solved_state_flattened:csdl.Variable = None
        self.state_scaler:np.ndarray = None
        self.stage_scaler:np.ndarray = None

class DynamicParameter(object):
    def __init__(self, name:str, parameter:csdl.Variable) -> None:
        self.name:str = name
        self.parameter:csdl.Variable = parameter
        self.given_shape = parameter.shape
        self.point_shape = parameter.shape[1:]
        self.interpolated_parameter:csdl.Variable = None

class States(object):
    def __init__(self) -> None:
        self.states:dict[str, State] = {}
        self.full_size:int = 0

    def add(
            self,
            name:str,
            initial_condition:csdl.Variable,
            output:bool,
            store_final:bool,
            initial_guess:np.ndarray,
            scaler
        ):

        if name in self.states:
            raise ValueError(f'state with name \'{name}\' already exists')
        self.states[name] = State(
            name,
            initial_condition,
            output,
            store_final,
            initial_guess,
            scaler,
        )

    def validate(self, num_times:int):
        if len(self.states) == 0:
            raise ValueError('no states added to the ODE problem')
        
        for name, state in self.states.items():
            state.solved_shape = (num_times,) + state.shape
            self.full_size += state.size

            # Check to make sure the inital guess is the correct shape
            if state.initial_guess is not None:
                if state.initial_guess.shape != (num_times-1,) + state.shape:
                    raise ValueError(f'Expected shape {(num_times-1,) + state.shape} for initial guess of state \'{name}\' but got {state.initial_guess.shape}')

    def compute_vector_constants(       
            self,
            glm_method,
            num_steps:int,
        )->None:
        import scipy.sparse as sp

        for name, state in self.states.items():
            # Creating GLM matrices corresponding to each state:
            A_kron = sp.kron(sp.csc_matrix(glm_method.A), sp.eye(
                state.size, format='csc'), format='csr')
            B_kron = sp.kron(sp.csc_matrix(glm_method.B), sp.eye(
                state.size, format='csc'), format='csr')
            U_kron = sp.kron(sp.csc_matrix(glm_method.U), sp.eye(
                state.size, format='csc'), format='csr')
            V_kron = sp.kron(sp.csc_matrix(glm_method.V), sp.eye(
                state.size, format='csc'), format='csr')

            state.constants['full_A'] = sp.kron(
                sp.eye(num_steps, format='csc'), A_kron, format='csc')
            state.constants['full_U'] = sp.kron(
                sp.eye(num_steps, n=num_steps+1, format='csc'), U_kron, format='csc')
            state.constants['full_B'] = sp.kron(sp.eye(
                num_steps + 1, n=num_steps, k=-1, format='csc'), B_kron, format='csc')
            state.constants['full_V'] = sp.kron(sp.eye(
                num_steps+1, k=-1, format='csc'), V_kron, format='csc')

            # Try to define initial guesses for collocation
            if state.initial_guess is not None:
                initial_guess = state.initial_guess
                initial_guess_stage = np.repeat(initial_guess, glm_method.num_stages, axis=0)

            elif state.initial_condition.value is not None:
                initial_guess = np.broadcast_to(
                    state.initial_condition.value, (num_steps,) + state.shape)
                initial_guess_stage = np.broadcast_to(
                    state.initial_condition.value, (num_steps*glm_method.num_stages,) + state.shape)
            else:
                initial_guess = np.ones((num_steps,) + state.shape)
                initial_guess_stage = np.ones((num_steps*glm_method.num_stages,) + state.shape)

            state.initial_guess_stage = initial_guess_stage
            state.initial_guess = initial_guess

    def compute_timemarching_constants(       
            self,
            glm_method,
        )->None:
        import scipy.sparse as sp

        # For state computation
        for name, state in self.states.items():
            state.constants['B_kron'] = sp.kron(sp.coo_matrix(
                glm_method.B), sp.eye(state.size, format='csc'), format='csc')
            state.constants['V_kron'] = sp.kron(sp.coo_matrix(
                glm_method.V), sp.eye(state.size, format='csc'), format='csc')

        # For stage computatation:
        if not glm_method.explicit:
            for name, state in self.states.items():
                state.constants['A_kron'] = sp.kron(sp.coo_matrix(
                    glm_method.A), sp.eye(state.size, format='csc'), format='csc')
                state.constants['U_kron'] = sp.kron(sp.coo_matrix(
                    glm_method.U), sp.eye(state.size, format='csc'), format='csc')

    def expand_step_vector(
            self,
            step_vector:csdl.Variable,
            num_steps:int,
            num_stages:int,
        ):
        for name, state in self.states.items():
            point_string = get_general_expand_string(state.shape)
            state.expanded_step_vector = step_vector.expand(
                (num_steps,num_stages) + state.shape,
                action = f'i->ij{point_string}',
            ).flatten()

    def expand_initial_conditions(
            self,
            num_times:int,
        ):
        for name, state in self.states.items():
            vectorized_initial_condition = np.zeros(num_times*state.size)
            vectorized_initial_condition = csdl.Variable(value = vectorized_initial_condition)
            state.vectorized_initial_condition = vectorized_initial_condition.set(csdl.slice[0:state.size], state.initial_condition.flatten())

    def build_implicit_variables(
            self,
            num_nodes:int,
        ):
        for name, state in self.states.items():
            # TODO: initial guess in value
            state.implicit_variable = csdl.ImplicitVariable(
                name=f'{name}_implicit_variable',
                shape = (num_nodes,) + state.shape, 
                value = state.initial_guess_stage,
            )

    def build_design_variables(
          self,
          glm_method,
          num_steps:int,
        ):
        for name, state in self.states.items():
            
            # Base scaler on initial guess
            if state.scaler is None:
                state.state_scaler = 1/np.average(np.abs(state.initial_guess)) if np.abs(np.average(state.initial_guess)) > 1e-7 else 1.0
                state.stage_scaler = 1/np.average(np.abs(state.initial_guess_stage)) if np.abs(np.average(state.initial_guess_stage)) > 1e-7 else 1.0
            else:
                state.state_scaler = state.scaler
                state.stage_scaler = state.scaler

            # first design variable for colloation based on the stages
            state.stage_dv = csdl.Variable(
                name=f'{name}_stage_dv',
                shape = (num_steps*glm_method.num_stages*state.size, ),
                value = state.initial_guess_stage.flatten(),
            )

            # second design variable for collocation based on the states
            state.state_dv = csdl.Variable(
                name=f'{name}_state_dv',
                shape = (num_steps*state.size, ),
                value = state.initial_guess.flatten(),
            )
            state.stage_dv.set_as_design_variable(scaler=state.stage_scaler)
            state.state_dv.set_as_design_variable(scaler=state.state_scaler)

            # Solved state that is given to the user
            solved_state = csdl.Variable(value = np.zeros((num_steps+1)*state.size))
            solved_state = solved_state.set(csdl.slice[0:state.size], state.initial_condition.flatten())
            solved_state = solved_state.set(csdl.slice[state.size:], state.state_dv)
            state.solved_state_flattened = solved_state
            state.solved_state = solved_state.reshape((num_steps+1,) + state.shape)

class DynamicParameters(object):
    def __init__(self) -> None:
        self.d_params:dict[str, DynamicParameter] = {}
        self.interpolated:bool = False

    def add(
            self,
            name:str,
            parameter:csdl.Variable,
        ):
        if name in self.d_params:
            raise ValueError(f'dynamic parameter with name \'{name}\' already exists')

        if len(parameter.shape) == 1:
            parameter = parameter.reshape((parameter.shape[0], 1))
        self.d_params[name] = DynamicParameter(name, parameter)

    def validate(self, num_times:int):
        for name, param in self.d_params.items():
            if param.parameter.shape[0] != num_times:
                raise ValueError(f'Expected first axis of dynamic parameter \'{name}\' to be size {num_times} but got {param.parameter.shape[0]}')
    
    def interpolate(
            self,
            glm_method,
            num_steps:int,
            flatten:bool = True,
        )->None:
        if self.interpolated:
            raise ValueError('Dynamic parameters already interpolated')
        # C = glm_method.C
        # print(C)
        n_stage = glm_method.num_stages
        num_times = num_steps + 1
        for name, d_param in self.d_params.items():
            # TODO: handle any shape of dynamic parameter
            point_string = get_general_expand_string(d_param.point_shape)
            dp_new = d_param.parameter.expand(
                (num_times, n_stage,) + d_param.point_shape,
                action = f'i{point_string}->ik{point_string}',
            )
            dp_new = dp_new.reshape((num_times*n_stage,) + d_param.point_shape)

            param_0 = dp_new[:-n_stage]
            param_1 = dp_new[n_stage:]

            # OLD:
            # c_tiled = np.tile(glm_method.C.flatten(),num_steps).reshape(num_steps*n_stage,1)
            # c_tiled_param = np.broadcast_to(c_tiled, (num_steps*n_stage,) + d_param.point_shape)
            
            # NEW:
            c_tiled = np.tile(glm_method.C.flatten(),num_steps).reshape((num_steps*n_stage,))
            c_tiled = np.expand_dims(c_tiled, axis = [1+i for i in range(len(d_param.point_shape))])
            c_tiled_param = c_tiled*np.ones((num_steps*n_stage,) + d_param.point_shape)

            interpolated_param = param_0*(1.0 - c_tiled_param) + param_1*c_tiled_param
            d_param.interpolated_parameter = interpolated_param

            if not flatten:
                d_param.interpolated_parameter = interpolated_param.reshape((num_steps, n_stage,) + d_param.point_shape)

class ProfileOutput(object):
    def __init__(self, name, shape) -> None:
        self.name:str = name
        self.point_shape:tuple[int] = shape
        self.solution:csdl.Variable = None

    def set_solution(self, solution:csdl.Variable, num_times:int):
        if solution.shape != (num_times,)+self.point_shape:
            raise ValueError(f'Expected shape {(num_times,)+self.point_shape} for profile output \'{self.name}\' but got {solution.shape}')
        self.solution = solution

class ProfileOutputs(object):
    def __init__(self) -> None:
        self.profile_outputs:dict[str, ProfileOutput] = {}
        self.exists:bool = False

    def add(self, name:str, shape:tuple[int]):
        self.profile_outputs[name] = ProfileOutput(name, shape)
        self.exists:bool = True

class FieldOutput(object):
    def __init__(self, name, shape) -> None:
        self.name = name
        self.point_shape = shape
        self.solution:csdl.Variable = None

    def set_solution(self, solution:csdl.Variable):
        if solution.shape != self.point_shape:
            raise ValueError(f'Expected shape {self.point_shape} for field output \'{self.name}\' but got {solution.shape}')
        self.solution = solution

class FieldOutputs(object):
    def __init__(self) -> None:
        self.field_outputs:dict[str, FieldOutput] = {}
        self.exists:bool = False

    def add(self, name:str, shape:tuple[int]):
        self.field_outputs[name] = FieldOutput(name, shape)
        self.exists:bool = True

class IntegratorOutputs(object):
    def __init__(self) -> None:
        self.states:dict[str, csdl.Variable] = {}
        self.final_states:dict[str, csdl.Variable] = {}
        self.profile_outputs:dict[str, csdl.Variable] = {}
        self.field_outputs:dict[str, csdl.Variable] = {}

    def __str__(self):
        prints = [
            ('States:', self.states),
            ('Final States:', self.final_states),
            ('Profile Outputs:', self.profile_outputs),
            ('Field Outputs:', self.field_outputs),
        ]

        string = '\nIntegrator outputs:\n'
        for name, vars in prints:
            string += f'  {name}\n'
            for key, value in vars.items():
                string += f'    {key}: {value.shape}\n'
        return string