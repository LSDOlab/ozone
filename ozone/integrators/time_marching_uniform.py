from ozone.integrators.time_marching import TimeMarching

import csdl_alpha as csdl
import numpy as np

class TimeMarchingUniform(TimeMarching):
    def integrate(self):
        
        # Process inputs
        self.dynamic_parameters.interpolate(
            self.method,
            self.num_steps,
            flatten=False,
        )
        # Build constant GLM matrices:
        self.states.compute_timemarching_constants(self.method)

        # Compute number of checkpoints automatically if not specified
        num_checkpoints = self.approach.num_checkpoints
        if num_checkpoints is None:
            num_checkpoints = int(self.num_steps**0.5)
        self.num_checkpoints:int = num_checkpoints # number of checkpoints --> number of times the inner loop will run
        self.num_outer_steps:int = self.num_checkpoints # number of outer steps, same as above.

        if not (self.num_steps % self.num_checkpoints == 0):
            raise ValueError(f'Number of timesteps ({self.num_steps}) must be divisible by number of checkpoints {self.num_checkpoints}')
        self.num_inner_steps:int = self.num_steps // self.num_checkpoints # number of steps in the inner loop
        inner_lower_indices = np.arange(num_checkpoints) * self.num_inner_steps

        # Need to reshape interpolated parameters: (num_steps, num_stages, ...) --> (num_outer, num_inner, num_stages, ...) to avoid dynamic slices
        for name, parameter in self.dynamic_parameters.d_params.items():
            parameter.interpolated_parameter = parameter.interpolated_parameter.reshape(self.num_outer_steps, self.num_inner_steps, self.method.num_stages, *parameter.point_shape)

        print('\nCheckpoint information:================')
        print('   # of points:       ', self.num_times)
        print('   # of steps:        ', self.num_steps)
        print('   # of checkpoints:  ', self.num_checkpoints)
        print('   # of inner iter:   ', self.num_inner_steps)
        print('   # of outer iter:   ', self.num_outer_steps)
        print('=======================================\n')

        # Outer loop of the checkpointing:
        with csdl.experimental.enter_loop([list(range(self.num_outer_steps)) ,list(inner_lower_indices)]) as loop_builder_outer:
            # Get inded variables
            outer_index, checkpoint_lower = loop_builder_outer.get_loop_indices()
            outer_index:csdl.Variable = outer_index
            checkpoint_lower:csdl.Variable = checkpoint_lower
            checkpoint_upper = checkpoint_lower + self.num_inner_steps

            # get current outer step sizes
            outer_step_vector:csdl.Variable = self.step_vector[checkpoint_lower:checkpoint_upper]
            
            # get current outer interpolated parameters 
            # ============================= old: ============================= : TODO: deprecate
            # lower_outer_param_index = checkpoint_lower * self.method.num_stages
            # upper_outer_param_index = checkpoint_upper * self.method.num_stages
            # outer_interp_params:dict[str,csdl.Variable] = {
            #     name:parameter.interpolated_parameter[lower_outer_param_index:upper_outer_param_index] for name, parameter in self.dynamic_parameters.d_params.items()
            # }

            # ============================= NEW: ============================= :
            outer_interp_params:dict[str,csdl.Variable] = {
                name:parameter.interpolated_parameter[outer_index] for name, parameter in self.dynamic_parameters.d_params.items()
            }

            # get current previous states:
            outer_previous_states = {
                name:loop_builder_outer.initialize_feedback(state.initial_condition) for name, state in self.states.states.items()
            }

            # check shapes:
            assert outer_step_vector.size == self.num_inner_steps
            for name, param in outer_interp_params.items():
                assert param.shape[0:2] == (self.num_inner_steps,self.method.num_stages)
            
            # Inner checkpointing loop:
            with csdl.experimental.enter_loop(self.num_inner_steps) as loop_builder_inner:
                inner_index = loop_builder_inner.get_loop_indices()

                # solve the integeration like normal:
                # - Current time step step size
                h = outer_step_vector[inner_index]
                
                # - Current time step dynamic parameters
                # ============================= old: ============================= : TODO: deprecate
                # lower_param_index = inner_index*self.method.num_stages
                # upper_param_index = lower_param_index + self.method.num_stages
                # current_params = {
                #     name:outer_param[lower_param_index:upper_param_index] for name, outer_param in outer_interp_params.items()
                # }

                # ============================= NEW: ============================= :
                current_params = {
                    name:outer_param[inner_index] for name, outer_param in outer_interp_params.items()
                }
                non_interp_params = {
                    name:parameter.parameter[checkpoint_lower+inner_index+1] for name, parameter in self.dynamic_parameters.d_params.items()
                }

                # - Previous time step state values
                previous_states = {
                    name:loop_builder_inner.initialize_feedback(outer_prev_state) for name, outer_prev_state in outer_previous_states.items()
                }

                # march time by one
                current_states, field_points, profile_points = self.march_time(
                    previous_states, 
                    h, 
                    current_params, 
                    non_interp_params, 
                    store_explicit_intermediate_states = False,
                )

                # build feedbacks
                for name, current_state in current_states.items():
                    loop_builder_inner.finalize_feedback(previous_states[name], current_state)

                # inner_index.print_on_update('   inner')

            # Accumulate profile/field outputs through the loop
            field_points = {field_name:loop_builder_inner.add_pure_accrue(field_points[field_name]) for field_name in field_points}
            profile_points = {profile_name:loop_builder_inner.add_stack(profile_points[profile_name]) for profile_name in profile_points}
            state_points = {name:loop_builder_inner.add_stack(current_states[name]) for name,state in self.states.states.items() if state.output}
            outer_current_states = {name:loop_builder_inner.add_output(current_states[name]) for name,state in self.states.states.items()}
            loop_builder_inner.finalize()

            # finalize feedbacks for outer loop:
            for name, outer_current_state in outer_current_states.items():
                loop_builder_outer.finalize_feedback(outer_previous_states[name], outer_current_state)

        # Post process outputs for the user specified outputs
        field_points:dict[str,csdl.Variable] = {field_name:loop_builder_outer.add_pure_accrue(field_points[field_name]) for field_name in field_points}
        profile_points:dict[str,csdl.Variable] = {profile_name:loop_builder_outer.add_stack(profile_points[profile_name]) for profile_name in profile_points}
        state_points:dict[str,csdl.Variable] = {name:loop_builder_outer.add_stack(state_points[name]) for name,state in self.states.states.items() if state.output}
        final_state_points:dict[str,csdl.Variable] = {name:loop_builder_outer.add_output(outer_current_states[name]) for name,state in self.states.states.items() if state.store_final}
        loop_builder_outer.finalize()

        # We need to reshape the stacks because they will have an extra dimensions from the double loop
        for profile_name, double_stacked_profiles in profile_points.items():
            profile_points[profile_name] = double_stacked_profiles.reshape((self.num_steps, *self.profile_outputs.profile_outputs[profile_name].point_shape))
        for state_name, double_stacked_states in state_points.items():
            state_points[state_name] = double_stacked_states.reshape((self.num_steps, *self.states.states[state_name].shape))

        # finalize outputs through the loop
        self.finalize_time_loop(
            fields = field_points,
            profiles =  profile_points,
            final_states = final_state_points,
            state_history = state_points,
        )


