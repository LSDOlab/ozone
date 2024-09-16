from ozone_alpha.integrators.vector_based import VectorBased
import csdl_alpha as csdl
import scipy.sparse.linalg as spln
import scipy.sparse as sp

class Collocation(VectorBased):

    def solve(self):
        # Compute Collocation specific constants
        for state_name, state in self.states.states.items():
            pass

        # Set up design variables:
        self.states.build_design_variables(
            self.method,
            self.num_steps,
        )

        # Evaluate the ODE:
        num_nodes=self.num_steps*self.method.num_stages
        ode_f = self.ode_func.evaluate(
            states = {
                name:state.stage_dv.reshape((num_nodes, *state.shape)) for name, state in self.states.states.items()
                },
            parameters = {
                name:parameter.interpolated_parameter for name, parameter in self.dynamic_parameters.d_params.items()
                },
            num_nodes=num_nodes,
        )
        self.initialize_outputs()

        # Compute the constraints of the collocation problem:
        for state_name, state in self.states.states.items():
            # Constants: scipy sparse matrices
            A = state.constants['full_A']
            B = state.constants['full_B']
            U = state.constants['full_U']
            V = state.constants['full_V']

            # Design variables from earlier
            Y_dv = state.stage_dv
            y_dv = state.solved_state_flattened

            # Evaluated ODE function
            fbar = ode_f.d_states[state_name].flatten()
            hF = state.expanded_step_vector*fbar

            c1 = csdl.sparse.matvec(A, hF) + csdl.sparse.matvec(U, y_dv) - Y_dv
            c2 = csdl.sparse.matvec(B, hF) + csdl.sparse.matvec(V, y_dv) - y_dv
            c2 = c2[state.size:]

            # Add the constraints to the solver
            c1.add_name(f'{state_name}_stage_c')
            c2.add_name(f'{state_name}_state_c')

            c1.set_as_constraint(upper = 0.0, lower = 0.0, scaler=state.stage_scaler)
            c2.set_as_constraint(upper = 0.0, lower = 0.0, scaler=state.stage_scaler)
