from ozone.integrators.integrator import IntegratorBase
import csdl_alpha as csdl

class VectorBased(IntegratorBase):
    
    def solve(self):
        raise NotImplementedError('solve method must be implemented in derived class')

    def integrate(self):

        # Compute constants
        self.states.compute_vector_constants(
            self.method,
            self.num_steps
        )

        # Process inputs
        self.dynamic_parameters.interpolate(
            self.method,
            self.num_steps,
        )
        self.states.expand_step_vector(
            self.step_vector,
            self.num_steps,
            self.method.num_stages,
        )
        self.states.expand_initial_conditions(self.num_times)
        
        # Set up the vectorized ODE
        self.solve()

        # compute profile/field outputs
        if self.profile_outputs.exists or self.field_outputs.exists:
            # evaluate the profile outputs
            outputs_f = self.ode_func.evaluate(
                states = {
                    name:state.solved_state for name, state in self.states.states.items()
                    },
                parameters = {
                    name:parameter.parameter for name, parameter in self.dynamic_parameters.d_params.items()
                    },
                num_nodes=self.num_times,
            )
            
            for profile_name, profile_output in self.profile_outputs.profile_outputs.items():
                profile_output.set_solution(outputs_f.profile_outputs[profile_name], num_times=self.num_times)

            for field_name, field_output in self.field_outputs.field_outputs.items():
                field_output.set_solution(csdl.sum(outputs_f.field_outputs[field_name], axes=(0,)))

        for state in self.states.states.values():
            if state.store_final:
                state.final_state = state.solved_state[-1]