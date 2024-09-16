from ozone.integrators.vector_based import VectorBased
import csdl_alpha as csdl

import scipy.sparse.linalg as spln
import scipy.sparse as sp

class PicardIteration(VectorBased):
    
    def solve(self):
        # Compute picard-iteration specific constants for
        for state_name, state in self.states.states.items():
            V_kron = sp.kron(sp.csc_matrix(self.method.V), sp.eye(
                state.size, format='csc'), format='csr')

            ImV_full = sp.eye((self.num_times)*state.size, format='csc') - sp.kron(
                sp.eye(self.num_times, k=-1, format='csc'), V_kron, format='csc')
            ImV_full_inv = spln.inv(ImV_full)
            state.constants['I_minus_V_inverse'] = ImV_full_inv
            state.constants['UI_minus_V_inverse'] = state.constants['full_U']@ImV_full_inv

        # Evaluate the ODE:
        self.states.build_implicit_variables(self.num_steps*self.method.num_stages)
        ode_f = self.ode_func.evaluate(
            states = {
                name:state.implicit_variable for name, state in self.states.states.items()
                },
            parameters = {
                name:parameter.interpolated_parameter for name, parameter in self.dynamic_parameters.d_params.items()
                },
            num_nodes=self.num_steps*self.method.num_stages,
        )
        self.initialize_outputs()

        # create nonlinear_solver
        solver = csdl.nonlinear_solvers.GaussSeidel(name = 'ozone_picard_solver', residual_jac_kwargs={'loop':False}, max_iter=2000)
        for state_name, state in self.states.states.items():
            # Constants
            UI_minus_V_inverse = state.constants['UI_minus_V_inverse']
            A = state.constants['full_A']
            B = state.constants['full_B']

            # Variables
            fbar = ode_f.d_states[state_name].flatten()
            hF = state.expanded_step_vector*fbar
            initial_condition = state.vectorized_initial_condition

            state.stages = A@hF + UI_minus_V_inverse@(B@hF+initial_condition)
            state.hF = hF

            # Define state-residual pair here:
            residual = state.implicit_variable - state.stages.reshape(state.implicit_variable.shape)
            solver.add_state(state.implicit_variable, residual, state_update=state.implicit_variable - residual) #TODO: add optional coefficient in front of residual
        solver.run()

        # Given the solved stages, compute the final states
        for state_name, state in self.states.states.items():
            # constants:
            I_minus_V_inverse = state.constants['I_minus_V_inverse']
            B = state.constants['full_B']

            initial_condition = state.vectorized_initial_condition

            # solve states
            hF = state.hF
            state.solved_state = (I_minus_V_inverse@(B@hF+initial_condition)).reshape(state.solved_shape)

        