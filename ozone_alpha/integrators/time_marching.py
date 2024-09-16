from ozone_alpha.integrators.integrator import IntegratorBase
import csdl_alpha as csdl
import numpy as np

class TimeMarching(IntegratorBase):
    def integrate(self):

        # Process inputs
        step_vector:csdl.Variable = self.step_vector
        self.dynamic_parameters.interpolate(
            self.method,
            self.num_steps,
            flatten = False,
        )
        # Build constant GLM matrices:
        self.states.compute_timemarching_constants(self.method)

        # perform integration
        with csdl.experimental.enter_loop(self.num_steps) as loop_builder:

            # Process inputs:
            # - Current time step step size
            current_timestep_index = loop_builder.get_loop_indices()
            h = step_vector[current_timestep_index]
            
            # - Current time step dynamic parameters
            interp_params = {
                name:parameter.interpolated_parameter[current_timestep_index] for name, parameter in self.dynamic_parameters.d_params.items()
            }
            non_interp_params = {
                name:parameter.parameter[current_timestep_index+1] for name, parameter in self.dynamic_parameters.d_params.items()
            }

            # - Previous time step state values
            previous_states = {
                name:loop_builder.initialize_feedback(state.initial_condition) for name, state in self.states.states.items()
            }

            # march time by one
            current_states, field_points, profile_points = self.march_time(previous_states, h, interp_params, non_interp_params)

            # build feedbacks
            for name, current_state in current_states.items():
                loop_builder.finalize_feedback(previous_states[name], current_state)

        # Post process outputs for the user specified outputs
        field_points:dict[str,csdl.Variable] = {field_name:loop_builder.add_pure_accrue(field_points[field_name]) for field_name in field_points}
        profile_points:dict[str,csdl.Variable] = {profile_name:loop_builder.add_stack(profile_points[profile_name]) for profile_name in profile_points}
        state_points:dict[str,csdl.Variable] = {name:loop_builder.add_stack(current_states[name]) for name,state in self.states.states.items() if state.output}
        final_state_points:dict[str,csdl.Variable] = {name:loop_builder.add_output(current_states[name]) for name,state in self.states.states.items() if state.store_final}
        loop_builder.finalize(stack_all=True)

        # finalize outputs through the loop
        self.finalize_time_loop(
            fields = field_points,
            profiles =  profile_points,
            final_states = final_state_points,
            state_history = state_points,
        )

    def march_time(
            self,
            previous_states:dict[str,csdl.Variable],
            h:csdl.Variable,
            current_params:dict[str,csdl.Variable],
            non_interp_params:dict[str,csdl.Variable],
            store_explicit_intermediate_states:bool = True,
        )->tuple[dict[str,csdl.Variable],dict[str,csdl.Variable],dict[str,csdl.Variable]]:

        # Compute stage values
        # --  if implicit method, solve using linear solver
        # --  if explicit method, compute stage values directly
        F_current = self.compute_stage(
            prev_state = previous_states,
            current_step = h,
            current_interpolated_params = current_params,
            store_explicit_intermediate_states = store_explicit_intermediate_states,
        )
        self.initialize_outputs()

        # Compute state value from stages
        current_states = self.compute_state(
            F_current = F_current,
            prev_state = previous_states,
            h = h,
        )

        # Run post processing to compute field/profile outputs
        # compute profile/field outputs
        field_points, profile_points = self.compute_outputs(
            states = current_states,
            dynamic_parameters = non_interp_params,
        )

        return current_states, field_points, profile_points
            
    def finalize_time_loop(
            self,
            fields:dict[str,csdl.Variable],
            profiles:dict[str,csdl.Variable],
            final_states:dict[str,csdl.Variable],
            state_history:dict[str,csdl.Variable],
        )->None:
        # compute profile/field outputs for initial conditions.
        initial_field_output, initial_profile_output = self.compute_outputs(
            {name:state.initial_condition for name, state in self.states.states.items()},
            {name:d_param.parameter[0] for name,d_param in self.dynamic_parameters.d_params.items()},
        )

        # Set accumulation solutions if specified and append/add to the initial conditions which aren't computed in the loop
        for field_name, field_output in self.field_outputs.field_outputs.items():
            field_output.set_solution(fields[field_name]+initial_field_output[field_name])
        
        for profile_name, profile_output in self.profile_outputs.profile_outputs.items():
            profile_output.set_solution(csdl.concatenate((initial_profile_output[profile_name].reshape(1,*profile_output.point_shape), profiles[profile_name])), num_times=self.num_times)

        # Set state solutions if specified as well. Also process with initial conditions if needed
        for state_name, state in self.states.states.items():
            if state.output:
                state.solved_state = csdl.concatenate((state.initial_condition.reshape(1,*state.shape), state_history[state_name]))
            if state.store_final:
                state.final_state = final_states[state_name]

    def compute_outputs(
            self,
            states:dict[str,csdl.Variable],
            dynamic_parameters:dict[str,csdl.Variable],
            )->tuple[dict[str,csdl.Variable],dict[str,csdl.Variable]]:

        if self.profile_outputs.exists or self.field_outputs.exists:
            outputs_f = self.ode_func.evaluate(
                states = {name:state.reshape(1,*state.shape) for name,state in states.items()},
                parameters = {name:dp.reshape(1,*dp.shape) for name,dp in dynamic_parameters.items()},
                num_nodes=1,
            )

            field_points = {field_name:outputs_f.field_outputs[field_name][0] for field_name in self.field_outputs.field_outputs}
            profile_points = {profile_name:outputs_f.profile_outputs[profile_name][0] for profile_name in self.profile_outputs.profile_outputs}
            return field_points, profile_points
        else:
            return {}, {}

    def compute_stage(
            self,
            prev_state:dict[str,csdl.Variable],
            current_step:csdl.Variable,
            current_interpolated_params:dict[str,csdl.Variable],
            store_explicit_intermediate_states:bool,
            )->dict[str,csdl.Variable]:

        num_stages = self.method.num_stages
        states = self.states.states

        if self.method.explicit: # Explicit time-marching
            # pre-allocate all stage values. TODO: this can be done without feedbacks?? (to remove stacking)
            prev_F = {name:csdl.Variable(value=np.zeros((num_stages, *state.shape))) for name,state in states.items()}
            
            # compute A matrix
            Ah = current_step*csdl.Variable(value=self.method.A)
            U = csdl.Variable(value=self.method.U)

            with csdl.experimental.enter_loop(num_stages) as stage_loop:
                # current stage index
                i = stage_loop.get_loop_indices()

                # compute stage Y
                current_stage = {}
                for name,state in states.items():
                    Y = U[i]*prev_state[name]
                    prev_F[name] = stage_loop.initialize_feedback(prev_F[name])
                    for ii in range(num_stages):
                        Y = Y + Ah[i,ii]*prev_F[name][ii]
                    current_stage[name] = Y.reshape(1, *state.shape)
                
                # compute stage F
                outputs_f = self.ode_func.evaluate(
                    states = current_stage,
                    parameters = {name:inter_param[i].reshape(1,*inter_param.shape[1:]) for name,inter_param in current_interpolated_params.items()},
                    num_nodes=1,
                )

                # fill in the stage values
                for name,state in states.items():
                    new_f = prev_F[name].set(csdl.slice[i], outputs_f.d_states[name].reshape(state.shape))
                    stage_loop.finalize_feedback(prev_F[name], new_f)
                    prev_F[name] = new_f
            stage_loop.finalize(stack_all=store_explicit_intermediate_states) # finish loop. prev_F should be the only outputs
            F_current = prev_F
        else: # Implicit time-marching:
            # R(Y) = Y - A*h*F(Y, pd) - U*y_previous
            # 1) Prep: pre-compute A (constant sparse) and U*y_previous 
            # 2) Initialize implicit state variable Y
            # 3) Compute F(Y, pd)
            # 4) R(Y) = Y - A*(h*F(Y, pd)) - U*y_previous # TODO:A*h can be precomputed once we have sparse arrays
            # 5) Solve R(Y) = 0 using a nonlinear solver
            # 6) Return F_current = F(Y, pd)
            
            # 1)
            
            # 2)
            Y_current:dict[str,csdl.Variable] = {}
            for name,state in states.items():
                Y:csdl.ImplicitVariable = csdl.ImplicitVariable(
                    name = f'{state.name}_implicit',
                    value=np.zeros((num_stages*state.size)),
                )
                Y_current[name] = Y
            
            # 3)
            outputs_f = self.ode_func.evaluate(
                states = {name:Yc.reshape(num_stages, *states[name].shape) for name, Yc in Y_current.items()},
                parameters = current_interpolated_params,
                num_nodes=num_stages,
            )

            # 4)
            nl_solver = csdl.nonlinear_solvers.Newton(
                'ozone_stage_solver',
                residual_jac_kwargs={'loop': False},
                tolerance = 1e-7,
            )
            # nl_solver = csdl.nonlinear_solvers.Newton('ozone_stage_solver', residual_jac_kwargs={'loop': True})
            # nl_solver = csdl.nonlinear_solvers.GaussSeidel('ozone newton solver',max_iter=2)
            for name,state in states.items():
                # constants
                h = current_step
                A = state.constants['A_kron']
                U = state.constants['U_kron']

                # ODE state/stages variables
                F = outputs_f.d_states[name].flatten()
                y = prev_state[name].flatten()

                # Compute
                R = Y_current[name] - A@(h*F) - U@y

                # Finalize solver
                nl_solver.add_state(Y_current[name], R, initial_value=U@y)
            
            nl_solver.run()

            # Store F_current for next step
            F_current = {name:outputs_f.d_states[name] for name in states}

        return F_current
    
    def compute_state(
            self,
            F_current:dict[str,csdl.Variable],
            prev_state:dict[str,csdl.Variable],
            h:csdl.Variable,
        )->dict[str,csdl.Variable]:

        outs:dict[str,csdl.Variable] = {}
        for name, f_current in F_current.items():
            state = self.states.states[name]
            constants = state.constants
            outs[name] = (constants['B_kron'] @ f_current.flatten())*h + constants['V_kron'] @ prev_state[name].flatten()
            outs[name] = outs[name].reshape(state.shape)
        return outs