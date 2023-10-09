# from petsc4py import PETSc
import warnings
# import cProfile
# import pstats
import time
from ozone.classes.integrators.IntegratorBase import IntegratorBase
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.linalg as ln
import numpy as np
import matplotlib.pyplot as plt
from ozone.classes.ODEModelTM import ODEModelTM


class TimeMarching(IntegratorBase):
    """
    TimeMarching is a child class of the IntegratorBase class.
    The TimeMarching class performs all calculations for the time-marching integration approach.
    """

    def post_setup_init(self):
        """
        post_setup_init() is run after the "setup" method and called when integrator system is created.
        All initializing operations that need to be done after "setup" is done here.

        Returns
        -------
            tuple
                Tuple of the number of nodes for ODE system, and number of nodes for profile output. number
                of nodes for ODE system = number of stages and number of nodes for profile out = 1 as it is
                not vectorized.
        """

        # IntegratorBase's post_setup_init is run first. Everything after this line is specific only to time-marching
        super().post_setup_init()

        # time-marching specific operations after here:
        # REVERSE JACVEC SETUP
        # Initializing JVP d_out
        self.d_out_num = 0
        for key in self.profile_output_dict:
            self.d_out_num += self.profile_output_dict[key]['num']
        for key in self.field_output_dict:
            self.d_out_num += self.field_output_dict[key]['num']
        self.d_out = np.zeros(self.d_out_num)

        # State Settings:
        self.total_numstagestate = 0
        for key in self.state_dict:
            sd = self.state_dict[key]

            # For each state, define constants:
            # num_stage_state = Number of stages times number of states
            self.state_dict[key]['num_stage_state'] = len(
                self.GLM_A) * sd['num']

            # shape_stage = shape of the full state_stage (aka tuple of num_stage_state)
            self.state_dict[key]['shape_stage'] = (sd['num_stage_state'],)

            # stage_ind = indices of concatenated states
            self.state_dict[key]['stage_ind'] = (
                self.total_numstagestate, self.total_numstagestate + sd['num_stage_state'])
            self.total_numstagestate += sd['num_stage_state']

            # set vectorized shape with number of nodes
            if type(self.state_dict[key]['shape']) == int:
                if self.state_dict[key]['shape'] == 1:
                    self.state_dict[key]['nn_shape_profile'] = (1,)
                else:
                    self.state_dict[key]['nn_shape_profile'] = (1, sd['shape'])
            elif type(self.state_dict[key]['shape']) == tuple:
                self.state_dict[key]['nn_shape_profile'] = (1,) + sd['shape']

            # Individual A, B, U, and V matrices
            self.state_dict[key]['A_kron'] = sp.kron(sp.coo_matrix(
                self.GLM_A), sp.eye(sd['num'], format='csc'), format='csc')
            # May be used for vector matrix multiplication is summation JVP for parameters?
            self.state_dict[key]['A_kron_csr'] = sp.kron(sp.coo_matrix(
                self.GLM_A), sp.eye(sd['num'], format='csc'), format='csr')
            self.state_dict[key]['U_kron'] = sp.kron(sp.coo_matrix(
                self.GLM_U), sp.eye(sd['num'], format='csc'), format='csc')
            self.state_dict[key]['B_kron'] = sp.kron(sp.coo_matrix(
                self.GLM_B), sp.eye(sd['num'], format='csc'), format='csc')
            self.state_dict[key]['V_kron'] = sp.kron(sp.coo_matrix(
                self.GLM_V), sp.eye(sd['num'], format='csc'), format='csc')

            # Individual precomputed transposed A, B, U, and V matrices
            self.state_dict[key]['A_kronT'] = self.state_dict[key]['A_kron'].transpose()
            self.state_dict[key]['U_kronT'] = self.state_dict[key]['U_kron'].transpose()
            self.state_dict[key]['B_kronT'] = self.state_dict[key]['B_kron'].transpose()
            self.state_dict[key]['V_kronT'] = self.state_dict[key]['V_kron'].transpose()

            # Identity matrix of full stage_state
            self.state_dict[key]['full_eye'] = sp.eye(
                sd['num_stage_state'], format='csc')

        self.stage_error_list = np.ones(len(self.state_dict))

        # STILL WIP: settings for explicit GLM method
        # ------------------------
        # self.explicit = False
        # ------------------------
        if self.explicit:
            self.A_rows = []
            self.U_rows = []

            for i in range(self.num_stages):
                # add rows of GLM A and B
                if i > 0:
                    self.A_rows.append(self.GLM_A[i, 0:i].flatten())
                else:
                    self.A_rows.append([])
                self.U_rows.append(self.GLM_U[i, :].flatten())

            self.Ah = np.copy(self.A_rows)

            # initialize 'f current' for each state
            for key in self.state_dict:
                sd = self.state_dict[key]
                sd['Yeval_current'] = np.ones(sd['shape_stage'])
                temp_list = []
                for i in range(self.num_stages):
                    temp_list.append((i*sd['num'], (i+1)*sd['num']))

                sd['psi_indices'] = tuple(temp_list)

            # For performance, precompute everything you can:
            self.explicit_tools = {}
            # tuple of reverse stage iteration
            self.explicit_tools['rev_stage_iter'] = tuple(reversed(range(self.num_stages)))

            # Preallocated psi_A
            self.explicit_tools['psi_A_temp'] = [None] * self.num_stages

            # Preallocate indices
            self.explicit_tools['stage_index_list'] = [None]*self.num_stages

            # Next two blocks of code gives a tuple of ONLY non-zero indices of the A matrix elements
            # to prevent unnecessarily looping through the zeros
            for s_num in self.explicit_tools['rev_stage_iter']:
                self.explicit_tools['stage_index_list'][s_num] = []
                for k in np.arange(self.num_stages-1, s_num, -1, dtype=int):
                    if self.Ah[k][s_num] != 0.0:
                        self.explicit_tools['stage_index_list'][s_num].append(k)
                self.explicit_tools['stage_index_list'][s_num] = tuple(self.explicit_tools['stage_index_list'][s_num])
            self.explicit_tools['stage_index_list'] = tuple(self.explicit_tools['stage_index_list'])

            self.explicit_tools['stage_index_list_fwd'] = [None]*self.num_stages
            for s_num in range(self.num_stages):
                self.explicit_tools['stage_index_list_fwd'][s_num] = []
                for k in range(s_num):
                    if self.Ah[s_num][k] != 0.0:
                        self.explicit_tools['stage_index_list_fwd'][s_num].append(k)
                self.explicit_tools['stage_index_list_fwd'][s_num] = tuple(self.explicit_tools['stage_index_list_fwd'][s_num])
            self.explicit_tools['stage_index_list_fwd'] = tuple(self.explicit_tools['stage_index_list_fwd'])

        self.f_list = []
        for key in self.state_dict:
            sd = self.state_dict[key]
            self.f_list.append(sd['f_name'])
        self.f_list = tuple(self.f_list)

        self.of_list = []
        self.wrt_list = []
        for key in self.f2s_dict:
            self.of_list.append(key)
            self.wrt_list.append(self.f2s_dict[key])
        for key in self.parameter_dict:
            self.wrt_list.append(key)
        self.of_list = tuple(self.of_list)
        self.wrt_list = tuple(self.wrt_list)

        # Return number of nodes for ODE and stages respectively.
        if self.explicit:
            return 1, 1
        else:
            return self.num_stages, 1

    def setup_integration(self):
        """
        Method that gets called before each integration (for checkpointing)
        """

        self.time_vector_full = np.zeros((self.num_steps+1))
        if self.times['type'] == 'step_vector':
            self.time_vector_full[1: self.num_steps +
                                  1] = np.cumsum(self.times['val'])
            self.h_vector_full = self.times['val']

    def integrate_ODE(self):
        """
        Integrates the ODE from the first timestep to the last timestep.
        """
        integration_settings = {
            'integration_type': 'FWD and REV',
            'state_IC': self.IC_dict,
            't_index_end': None,
            't_index_start': None}

        # ===UNCOMMENT FOR PROFILING===
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(self.integrate_ODE_phase)
        # lp_wrapper(integration_settings)
        # lp.print_stats()
        # ===UNCOMMENT FOR PROFILING===
        self.integrate_ODE_phase(integration_settings)

    def compute_JVP(self, d_inputs_in, d_outputs_in):
        """
        Compute Jacobian Vector product of the ODE from the last timestep to the first timestep.

        Parameters
        ----------
            d_inputs_in:

            d_outputs_in:
        """
        # ===UNCOMMENT FOR PROFILING===
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(self.compute_JVP_phase)
        # lp_wrapper(d_inputs_in, d_outputs_in)
        # lp.print_stats()
        # ===UNCOMMENT FOR PROFILING===
        return self.compute_JVP_phase(d_inputs_in, d_outputs_in)

    # Main Integration Method
    def integrate_ODE_phase(self, settings):
        """
        Integrates the ODE over a time_interval.

        Parameters
        ----------
            settings: Dict
                Dictionary containing settings:

            t_index_start: int
                The time index to start the integration process.

            t_index_end: int
                The time index to end the integration process.

            state_IC:
                The initial condition of the integration phase ie. state at the time index t_index_start

        """

        # Integration settings. This is needed as integration with checkpointing is slightly different than without it.
        integration_type = settings['integration_type']
        state_IC = settings['state_IC']
        t_index_end = settings['t_index_end']
        t_index_start = settings['t_index_start']
        start_total = time.time()

        # Different values are stored or discarded depending on the type of integration
        # Timemarching without checkpointing
        if integration_type == 'FWD and REV':
            # Depending on integration type, set boolean to tell integrator what to store

            # If true, compute and store field/profile outputs
            store_outputs = True

            # If true, store the state vectors
            store_states = True

            # if true, store the jacobians for all values
            store_jac = True

            # If true, store checkpoints
            store_checkpoints = False

            # Create time vector
            self.setup_integration()

        # Timemarching with checkpoint (Forward)
        elif integration_type == 'FWD':
            store_outputs = True
            store_states = False
            store_jac = False
            store_checkpoints = True

            # Store first checkpoint: The initial condition
            for key in self.checkpoints[self.num_checkpoints-1]['checkpoint_snapshot-']:
                sd = self.state_dict[key]
                snapshot = (self.IC_dict[sd['IC_name']]
                            ['val']).reshape((sd['num'],))
                self.checkpoints[self.num_checkpoints -
                                 1]['checkpoint_snapshot-'][key] = snapshot

            # The checkpoint list index that decreases everytime a checkpoint is stored
            checkpoint_index = self.num_checkpoints-2

            self.setup_integration()

        # Timemarching with checkpoint (Reverse)
        elif integration_type == 'REV':
            store_outputs = False
            store_states = True
            store_jac = False
            store_checkpoints = False

        self.trm = 0.
        # if time range isnt specified, just solve the entire ODE, calculate outputs = True
        if (t_index_start == None) or (t_index_end == None):
            t_index_start = 0
            t_index_end = self.num_steps+1

        if self.display != None:
            print('Integrating ODE ... ')

        # "IC's" for checkpointing
        phase_initial_condition_dict = state_IC

        time_vector = self.time_vector_full[t_index_start: t_index_end]
        h_vector = self.h_vector_full[t_index_start: t_index_end]
        numtimes = len(time_vector)-1

        # TIME ZERO OPERATIONS
        # Setting up storage vectors for each state:
        for key in self.state_dict:
            sd = self.state_dict[key]

            if store_states == True:
                self.state_dict[key]['y_storage'] = np.zeros(
                    (sd['num'], numtimes+1))
                self.state_dict[key]['Yeval_full'] = np.zeros(
                    (sd['num_stage_state'], numtimes+1))
                self.state_dict[key]['Y_prime_full'] = {}

                # Store initial conditions
                if integration_type == 'FWD and REV':
                    self.state_dict[key]['y_storage'][:, [
                        0]] = phase_initial_condition_dict[sd['IC_name']]['val'].reshape(sd['num'], 1)

                if integration_type == 'REV':
                    self.state_dict[key]['y_storage'][:, [
                        0]] = phase_initial_condition_dict[key].reshape(sd['num'], 1)

            if store_jac == True:
                # Y_prime for initial condition is not used in JVP
                for s_key in self.state_dict:
                    self.state_dict[key]['Y_prime_full'][s_key] = []
                    # if t_index_start == 0:
                    self.state_dict[key]['Y_prime_full'][s_key].append([])

                # Initializing df/dp storage:
                if self.param_bool == True:
                    self.state_dict[key]['df_dp'] = {}
                    self.state_dict[key]['df_dp+'] = {}

                    for p_key in self.parameter_dict:
                        self.state_dict[key]['df_dp'][p_key] = []
                        # if t_index_start == 0:
                        self.state_dict[key]['df_dp'][p_key].append([])

                        self.state_dict[key]['df_dp+'][p_key] = []
                        # if t_index_start == 0:
                        self.state_dict[key]['df_dp+'][p_key].append([])
            # Initial Condition:

            if integration_type == 'FWD and REV':
                self.state_dict[key]['y_previous'] = phase_initial_condition_dict[sd['IC_name']]['val'].reshape(
                    (sd['num'],))
            if integration_type == 'FWD':
                self.state_dict[key]['y_previous'] = phase_initial_condition_dict[sd['IC_name']]['val'].reshape(
                    (sd['num'],))
            if integration_type == 'REV':
                self.state_dict[key]['y_previous'] = phase_initial_condition_dict[key].reshape(
                    (sd['num'],))

            # Setting initial guess for stages Y_current:
            self.state_dict[key]['Y_current'] = np.ones(sd['shape_stage'])
        # setting time indices
        time_now_index = t_index_start

        # Setting Static Parameters:
        param_set = {}
        for key in self.parameter_dict:
            if self.parameter_dict[key]['dynamic'] == False:
                param_set[key] = self.parameter_dict[key]['val']
        self.ode_system.set_vars(param_set)

        # Profile/Field Outputs for initial conditions
        if store_outputs == True:
            # Need to make sure IC's are right for checkpointing in the future.
            if self.profile_outputs_bool == True:
                self.profile_outputs_system.set_vars(param_set)

                run_dict = {}
                output_vals = []
                for key in self.profile_output_dict:
                    output_vals.append(key)

                for state_name in self.state_dict:
                    temp = np.empty(self.state_dict[state_name]['nn_shape_profile'])
                    temp[0] = phase_initial_condition_dict[self.state_dict[state_name]['IC_name']]['val']
                    run_dict[state_name] = temp

                # Updating Dynamic Parameters:
                for key in self.parameter_dict:
                    if self.parameter_dict[key]['dynamic'] == True:
                        temp = np.array([self.parameter_dict[key]['val'][time_now_index].reshape(self.parameter_dict[key]['shape'])])
                        run_dict[key] = temp

                P = self.profile_outputs_system.run_model(
                    run_dict, output_vals)

            for key in self.profile_output_dict:
                self.profile_output_dict[key]['val'] = np.zeros(
                    self.profile_output_dict[key]['shape'])
                self.profile_output_dict[key]['val'][0] = P[key][0]
            for key in self.field_output_dict:
                self.field_output_dict[key]['val'] = np.zeros(
                    self.field_output_dict[key]['shape'])
                self.field_output_dict[key]['val'] += self.field_output_dict[key]['coefficients'][0] * \
                    phase_initial_condition_dict[self.state_dict[self.field_output_dict[key]
                                                                 ['state_name']]['IC_name']]['val']

            for key in self.state_output_tuple:
                sd = self.state_dict[key]
                sd['y_out'] = np.zeros(sd['output_shape'])
                sd['y_out'][0] = phase_initial_condition_dict[sd['IC_name']]['val'].reshape(sd['shape'])

        # TIME MARCHING LOOP------:
        for t in range(numtimes):
            start = time.time()

            # Current time
            time_now = time_vector[t+1]
            h = h_vector[t]
            time_now_index += 1

            # Updating Dynamic Parameters:
            param_set = {}
            for key in self.parameter_dict:
                if self.parameter_dict[key]['dynamic'] == True:
                    temp = self.parameter_dict[key]['val_nodal'][time_now_index-1].reshape(self.parameter_dict[key]['nn_shape'])
                    param_set[key] = temp

            # Set interpolated vectorized parameters
            if not self.explicit:
                self.ode_system.set_vars(param_set)

            # Main Integration Calculation:
            start_s = time.time()

            # -------------------------- MAIN INTEGRATION CALCULATIONS --------------------------:

            if self.explicit:
                # === UNCOMMENT FOR PROFILING ===
                # from line_profiler import LineProfiler
                # lp = LineProfiler()
                # lp_wrapper = lp(self.compute_stage_explicit)
                # lp_wrapper(t, h, store_jac)
                # lp.print_stats()
                # === UNCOMMENT FOR PROFILING ===
                self.compute_stage_explicit(time_now_index, h, store_jac)  # Part a and b (explicit)
            else:
                # === UNCOMMENT FOR PROFILING ===
                # from line_profiler import LineProfiler
                # lp = LineProfiler()
                # lp_wrapper = lp(self.compute_stage)
                # lp_wrapper(t, h, store_jac)
                # lp.print_stats()
                # === UNCOMMENT FOR PROFILING ===
                self.compute_stage(t, h, store_jac)  # Part a
                self.evaluate_stage(time_now_index)  # Part b
            end_s = time.time()
            self.compute_state(h)  # Part C
            # -------------------------- MAIN INTEGRATION CALCULATIONS --------------------------:

            # Storage for JVP:
            # if self.recorder is not None:
            #     writer_dict = {}

            for key in self.state_dict:
                sd = self.state_dict[key]

                if store_states == True:
                    self.state_dict[key]['Yeval_full'][:, t+1] = sd['Yeval_current']
                    self.state_dict[key]['y_storage'][:, t+1] = sd['y_current']
                self.state_dict[key]['y_previous'] = sd['y_current']

            #     if self.recorder is not None:
            #         writer_dict[key] = sd['y_current']

            # # print(self.recorder)
            # if self.recorder is not None:
            #     # print(writer_dict)
            #     self.recorder(writer_dict, 'ozone')

            # Profile/Field/State Outputs for current time step
            if store_outputs == True:
                if self.profile_outputs_bool == True:
                    run_dict = {}
                    output_vals = []
                    for key in self.profile_output_dict:
                        output_vals.append(key)

                    for state_name in self.state_dict:
                        temp = np.empty(self.state_dict[state_name]['nn_shape_profile'])
                        temp[0] = self.state_dict[state_name]['y_current'].reshape(self.state_dict[state_name]['shape'])
                        run_dict[state_name] = temp

                    # Updating Dynamic Parameters:
                    for key in self.parameter_dict:
                        if self.parameter_dict[key]['dynamic'] == True:
                            temp = np.array([self.parameter_dict[key]['val'][time_now_index].reshape(self.parameter_dict[key]['shape'])])
                            run_dict[key] = temp

                    trm = time.time()
                    P = self.profile_outputs_system.run_model(
                        run_dict, output_vals)
                    self.trm += (time.time() - trm)

                for key in self.profile_output_dict:
                    self.profile_output_dict[key]['val'][time_now_index] = P[key][0]
                for key in self.field_output_dict:
                    state_name = self.field_output_dict[key]['state_name']
                    self.field_output_dict[key]['val'] += self.field_output_dict[key]['coefficients'][time_now_index] * \
                        self.state_dict[state_name]['y_current'].reshape(
                            self.state_dict[state_name]['shape'])
                for key in self.state_output_tuple:
                    sd = self.state_dict[key]
                    self.state_dict[key]['y_out'][time_now_index] = sd['y_current'].reshape(sd['shape'])

            # Dynamic plotting if requested:
            if self.visualization == 'during':
                self.ongoingplot
                plt.clf()
                for key in self.state_dict:
                    for i in range(self.state_dict[key]['num']):
                        plt.plot(
                            time_vector[0:time_now_index+1], self.state_dict[key]['y_storage'][i, 0:time_now_index+1])
                plt.xlabel('Time')
                plt.ylabel('states')
                plt.grid(True)
                plt.draw()
                plt.pause(0.00001)

            # Store checkpoints if this integration process is FWD intergration of checkpointing scheme
            if store_checkpoints == True:

                # Check if we store this current state as a checkpoint
                if time_now_index == self.checkpoint_indices[checkpoint_index]:

                    # Store states
                    for key in self.checkpoints[checkpoint_index]['checkpoint_snapshot-']:
                        sd = self.state_dict[key]
                        self.checkpoints[checkpoint_index]['checkpoint_snapshot-'][key] = sd['y_current']

                    checkpoint_index = checkpoint_index - 1

            end = time.time()
            # print('Total Time: ',end - start)
            # print('section time ratio:',(end_s-start_s)/(end - start))
            # print('section time:', (end_s-start_s))

        # Plotting @ end if requested
        if self.visualization == 'end':
            self.ongoingplot
            plt.clf()
            for key in self.state_dict:
                for i in range(self.state_dict[key]['num']):
                    plt.plot(time_vector[0:time_now_index+1],
                             self.state_dict[key]['y_storage'][i, 0:time_now_index+1], label=key)
            plt.xlabel('Time')
            plt.ylabel('states')
            plt.legend()
            plt.grid(True)
            plt.draw()
            plt.pause(0.00001)

        end_total = time.time()
        if self.display != None:
            print('Function Evaluation Time: ', self.trm)
            # print('Section Time: ', end_s - start_s)
            print('Total Integration Time: ', end_total - start_total, '\n')
        return

    # Creating Methods for parts a, b, and c for ODE intergration
    # part A: Stage Compuation
    def compute_stage(self, t_now_index, h, store_jac):
        # Use Newton's Method to get F(Yn)
        start_f = time.time()

        # Initial iteration guess is previous stage
        for key in self.state_dict:
            sd = self.state_dict[key]
            sd['Y_iteration'] = sd['Y_current']
            sd['hAkron'] = h*sd['A_kron']
            sd['Uy'] = sd['U_kron']*(sd['y_previous'])
            # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@TEMPORARY: CHANGE LATER:@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # self.state_dict[key]['Y_iteration'] = np.ones(sd['shape_stage'])
            # # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@TEMPORARY: CHANGE LATER:@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        error = 5.
        iteration_num = 0

        # of_list, wrt_list for computing f and df/dy
        of_list = []
        wrt_list = []
        for key in self.f2s_dict:
            of_list.append(key)
            wrt_list.append(self.f2s_dict[key])
        run_dict = {}

        start_sec = time.time()

        # IMPLICIT METHOD:
        # Newton Iteration to compute stage
        # print()
        jac_dimensions = self.all_states_num*self.num_stages
        if jac_dimensions > 150:
            sparse_jac = True
        else:
            sparse_jac = False

        while error > self.error_tolerance:
            # print('ERROR: ', error, '/',self.error_tolerance)
            # start_stage = time.time()

            # RunModel and ComputingTotalDerivatives
            for key in self.state_dict:
                sd = self.state_dict[key]
                run_dict[key] = sd['Y_iteration'].reshape(sd['nn_shape'])
            trm = time.time()
            P = self.ode_system.run_model(run_dict, self.f_list)
            Pd = self.ode_system.compute_total_derivatives(of_list, wrt_list)

            self.trm += (time.time() - trm)

            # Newton's Iteration
            error_index = 0
            full_concatenated_residual = np.zeros(jac_dimensions)
            full_iter_matrix_list = []
            for key in self.state_dict:
                sd = self.state_dict[key]
                f_name = sd['f_name']
                full_iter_matrix_cur_row = []  # Will be used to build current block row

                # Compute residual
                Y_iteration_eval = P[f_name].reshape(sd['shape_stage'])
                Y_iteration = sd['Y_iteration']
                Residual = Y_iteration - (sd['hAkron'])*(Y_iteration_eval) - sd['Uy']  # R

                # Add to full residual vector
                full_concatenated_residual[sd['stage_ind'][0]:sd['stage_ind'][1]] = -Residual

                # Build Iteration matrix
                for key_wrt in self.state_dict:
                    swrt = self.state_dict[key_wrt]
                    Y_iteration_prime = Pd[f_name][key_wrt]
                    entry = -sd['hAkron'].dot(Y_iteration_prime)
                    if key_wrt == key:
                        entry = sd['full_eye'] + entry
                    # print(key,swrt)

                    if sparse_jac:
                        full_iter_matrix_cur_row.append(sp.csc_matrix(entry))
                    else:
                        if not sp.issparse(entry):
                            full_iter_matrix_cur_row.append((entry))
                        else:
                            full_iter_matrix_cur_row.append((entry.toarray()))

                full_iter_matrix_list.append(full_iter_matrix_cur_row)

            if sparse_jac:
                # Full iteration matrix for Newton's Method
                iter_matrix = sp.bmat(full_iter_matrix_list, format='csc')

                # solve for the steps
                concatenated_steps = spln.spsolve(iter_matrix, full_concatenated_residual)
                # print(iter_matrix.toarray())
                # exit()
            else:
                # Full iteration matrix for Newton's Method
                iter_matrix = np.block(full_iter_matrix_list)

                # solve for the steps
                concatenated_steps = np.linalg.solve(iter_matrix, full_concatenated_residual)
                # print(iter_matrix.toarray())
                # exit()

            error = 0.0
            for i, key in enumerate(self.state_dict):
                sd = self.state_dict[key]

                # Next step
                sk = concatenated_steps[sd['stage_ind'][0]:sd['stage_ind'][1]]

                # Next iteration
                Ynextiteration = sd['Y_iteration'] + sk
                # print(sk)

                # max error
                current_error = np.linalg.norm(sk)
                if current_error > error:
                    error = current_error

                # Next iteration
                sd['Y_iteration'] = Ynextiteration
            iteration_num += 1

            if iteration_num >= 100:
                print(f'Newtons Method did not converge to {self.error_tolerance} in 100 iterations. Current error is {error}.')
                break
            # exit()

            # # Setting Stage Iteration:
            # error_index = 0
            # for key in self.state_dict:
            #     start_stage = time.time()
            #     sd = self.state_dict[key]
            #     f_name = sd['f_name']

            #     # Setting F and dF/dY
            #     Y_iteration_eval = P[f_name].reshape(sd['shape_stage'])
            #     Y_iteration_prime = Pd[f_name][key]
            #     Y_iteration = sd['Y_iteration']

            #     # Calculating Derivatives
            #     Residual = Y_iteration - (sd['hAkron'])*(Y_iteration_eval) - sd['Uy']  # R
            #     # Y_new = Y_old - R/(pR/pY)

            #     # If Native System:
            #     if self.OStype == 'NS':
            #         partialtype = self.ode_system.partial_properties[f_name][key]['type']

            #         if partialtype == 'empty':
            #             Y_nextiteration = Y_iteration - Residual

            #         # Depending on sparse or not, calculate Y_nextiteration differently
            #         if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':

            #             if self.implicit_solver_fwd == 'direct':
            #                 Y_nextiteration = Y_iteration + spln.gmres(sd['full_eye'] - sd['hAkron'] * Y_iteration_prime, (-Residual))[0]  # Ynew

            #             elif self.implicit_solver_fwd == 'iterative':
            #                 sk_prev = np.ones(Residual.shape)
            #                 fpi_error = 1.0
            #                 fpi_iter = 0
            #                 while fpi_error > self.error_tolerance:
            #                     sk = sd['hAkron']*(Y_iteration_prime*sk_prev)-Residual
            #                     fpi_error = np.linalg.norm(sk_prev-sk)
            #                     fpi_iter += 1
            #                     sk_prev = sk
            #                     if fpi_iter > 10:
            #                         warnings.warn("iterative taking time to converge. An alternative is to use direct method: implicit_solver_fwd = 'direct'")
            #                 Y_nextiteration = Y_iteration + sk

            #         elif partialtype == 'std' or partialtype == 'cs_uc':
            #             Y_iteration_prime = Y_iteration_prime.reshape((sd['num_stage_state'], sd['num_stage_state']))

            #             if self.implicit_solver_fwd == 'direct':
            #                 temp = sd['full_eye'] - sd['hAkron'].dot(Y_iteration_prime)
            #                 Y_nextiteration = Y_iteration + ln.solve(temp, (-Residual))

            #             elif self.implicit_solver_fwd == 'iterative':
            #                 sk_prev = np.ones(Residual.shape)
            #                 fpi_error = 1.0
            #                 fpi_iter = 0
            #                 while fpi_error > self.error_tolerance:
            #                     sk = sd['hAkron'].dot(Y_iteration_prime.dot(sk_prev))-Residual
            #                     fpi_error = np.linalg.norm(sk_prev-sk)
            #                     fpi_iter += 1
            #                     sk_prev = sk
            #                     if fpi_iter > 10:
            #                         warnings.warn("iterative taking time to converge. An alternative is to use direct method: implicit_solver_fwd = 'direct'")
            #                 Y_nextiteration = Y_iteration + sk

            #     elif self.OStype == 'OM':
            #         if self.implicit_solver_fwd == 'direct':
            #             temp = sd['full_eye'] - sd['hAkron'].dot(Y_iteration_prime)
            #             Y_nextiteration = Y_iteration + ln.solve(temp, (-Residual))

            #         elif self.implicit_solver_fwd == 'iterative':
            #             sk_prev = np.ones(Residual.shape)
            #             fpi_error = 1.0
            #             fpi_iter = 0
            #             while fpi_error > self.error_tolerance:
            #                 sk = sd['hAkron'].dot(Y_iteration_prime.dot(sk_prev))-Residual
            #                 fpi_error = np.linalg.norm(sk_prev-sk)
            #                 sk_prev = sk
            #                 fpi_iter += 1
            #                 if fpi_iter > 10:
            #                     warnings.warn("iterative taking time to converge. An alternative is to use direct method: implicit_solver_fwd = 'direct'")
            #             Y_nextiteration = Y_iteration + sk
            #     # Error
            #     self.stage_error_list[error_index] = np.linalg.norm(
            #         Y_iteration-Y_nextiteration)
            #     error_index += 1
            #     self.state_dict[key]['Y_iteration'] = Y_nextiteration
            #     end_stage = time.time()
            #     # print('Stage Newton Section Ratio:',(end_s - start_s)/(end_stage - start_stage))
            #     # print('Stage Newton Section Time:', (end_s - start_s))
            #     # print(iteration_num, 'SNS Res:', (-Residual))

            # error = self.stage_error_list.max()
            # # print(iteration_num, self.stage_error_list)
            # # print('newtons error:', error)
            # # print('iteration_num',iteration_num, 'NEWTON ITR ERROR:', error)
            # iteration_num += 1
            # end_stage = time.time()
            # print(iteration_num)
            # print('Stage Newton Section Ratio:',(end_s - start_s)/(end_stage - start_stage))
        end_sec = time.time()
        #  ------------ Newton Iteration Finished --------------
        # Set Jacobian to zero for memory:
        Pd = 0
        Y_iteration_prime = 0.

        # Computing Total Derivatives after convergence:
        for key in self.state_dict:
            sd = self.state_dict[key]
            run_dict[key] = sd['Y_iteration'].reshape(sd['nn_shape'])
        self.ode_system.set_vars(run_dict)
        end_sec = time.time()

        # Compute derivatives of parameters as well
        for param in self.parameter_dict:
            wrt_list.append(param)

        # Compute derivatives
        trm = time.time()
        if self.OStype == 'OM':
            # If OpenMDAO ODE, we have to rerun the model whenever we want derivatives after setting variables
            self.ode_system.run_model({}, [])
        Pd = self.ode_system.compute_total_derivatives(of_list, wrt_list)
        self.trm += (time.time() - trm)
        # Setting and storing partials
        for key in self.state_dict:
            sd = self.state_dict[key]
            f_name = sd['f_name']

            # Setting F and dF/dY
            self.state_dict[key]['Y_current'] = sd['Y_iteration']

            # Store all jacobians
            if store_jac == True:
                for s_key in self.state_dict:

                    if self.OStype == 'NS':
                        partialtype = self.ode_system.partial_properties[f_name][s_key]['type']

                        if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                            self.state_dict[key]['Y_prime_full'][s_key].append(
                                Pd[f_name][s_key].copy())
                        else:
                            self.state_dict[key]['Y_prime_full'][s_key].append(
                                Pd[f_name][s_key])
                    elif self.OStype == 'OM':

                        if sp.issparse(Pd[f_name][s_key]):
                            self.state_dict[key]['Y_prime_full'][s_key].append(
                                Pd[f_name][s_key].copy())
                        else:
                            self.state_dict[key]['Y_prime_full'][s_key].append(
                                Pd[f_name][s_key])

        if store_jac == True:
            if self.param_bool == True:
                for key in self.state_dict:
                    sd = self.state_dict[key]
                    f_name = self.state_dict[key]['f_name']
                    for key_param in self.parameter_dict:
                        pd = self.parameter_dict[key_param]

                        # Need to reshape if dynamic parameters are dynamic
                        if pd['dynamic'] == False:
                            self.state_dict[key]['df_dp'][key_param].append(
                                Pd[f_name][key_param])

                        else:
                            df_dp = Pd[f_name][key_param].reshape(
                                (sd['num_stage_state'], self.num_stages*pd['num']))

                            # print('BEFORE TRANSFORM', df_dp.shape, )
                            if sp.issparse(df_dp):
                                df_dp_s = df_dp@pd['stage2state_transform_s']
                                df_dp_plus = df_dp * pd['stage2state_transform_s+']
                            else:
                                df_dp_s = df_dp.dot(
                                    pd['stage2state_transform_d'])
                                df_dp_plus = df_dp.dot(
                                    pd['stage2state_transform_d+'])

                            self.state_dict[key]['df_dp'][key_param].append(
                                df_dp_s)
                            self.state_dict[key]['df_dp+'][key_param].append(
                                df_dp_plus)
        end_f = time.time()
        return

    # part A: Stage computation explicit
    def compute_stage_explicit(self, t_now_index, h, store_jac):
        # Compute stage explicitly
        # Store each stage temporarily

        run_dict = {}

        # List to store stages for this timestep
        for key in self.state_dict:
            sd = self.state_dict[key]
            sd['k_temp'] = []

        # Precompute h*A:
        for i in range(len(self.A_rows)):
            if i > 0:
                self.Ah[i] = h*(self.A_rows[i])

        # Loop through number of stages:
        for s_num in range(self.num_stages):
            for key in self.state_dict:

                # For first stage, only a function of previous timestep
                sd = self.state_dict[key]
                if s_num == 0:
                    run_dict[key] = ((self.U_rows[0][0])*sd['y_previous']).reshape(sd['nn_shape_exp'])
                    continue

                # initialize stage shape as zeros. May need reshape in the future
                run_dict[key] = np.zeros(sd['num'])

                # for s_now in self.explicit_tools['stage_index_list_fwd'][s_num]:
                for s_now in range(s_num):
                    a_coeff = self.Ah[s_num][s_now]
                    run_dict[key] += a_coeff*sd['k_temp'][s_now]

                run_dict[key] += (self.U_rows[s_num][0])*sd['y_previous']
                run_dict[key] = run_dict[key].reshape(sd['nn_shape_exp'])

            # Setting parameters if dynamic
            for key in self.parameter_dict:
                if self.parameter_dict[key]['dynamic'] == True:
                    run_dict[key] = self.parameter_dict[key]['val_nodal'][t_now_index-1][s_num].reshape(self.parameter_dict[key]['nn_shape_exp'])

            # Compute F(Y) and dF(Y) for stage 's_num'
            trm = time.time()
            P = self.ode_system.run_model(run_dict, self.f_list)
            if store_jac:
                Pd = self.ode_system.compute_total_derivatives(self.of_list, self.wrt_list)
            self.trm += (time.time() - trm)

            # Store stage in list initialized before loop
            for key in self.state_dict:
                sd = self.state_dict[key]
                f_name = sd['f_name']
                sd['k_temp'].append(P[f_name].reshape(sd['num']).copy())

            # Store derivatives for later if required
            # if store_jac:
            if store_jac:

                # Storing State Jac:
                for s_of in self.state_dict:
                    sd = self.state_dict[s_of]
                    f_name = sd['f_name']
                    for s_wrt in self.state_dict:
                        # store derivatives as a list with index corresponding to stage number
                        if s_num == 0:
                            sd['Y_prime_full'][s_wrt].append([])

                        # State derivative jacobian
                        if self.OStype == 'OM':
                            if sp.issparse(Pd[f_name][s_wrt]):
                                sd['Y_prime_full'][s_wrt][-1].append((Pd[f_name][s_wrt].reshape((sd['num'], self.state_dict[s_wrt]['num']))).copy())
                            else: 
                                sd['Y_prime_full'][s_wrt][-1].append((Pd[f_name][s_wrt].reshape((sd['num'], self.state_dict[s_wrt]['num']))))
                        else:  # 'NS'
                            partialtype = self.ode_system.partial_properties[f_name][s_wrt]['type']

                            if partialtype == 'empty':
                                continue

                            if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                                sd['Y_prime_full'][s_wrt][-1].append((Pd[f_name][s_wrt].reshape((sd['num'], self.state_dict[s_wrt]['num']))).copy())
                            else:
                                sd['Y_prime_full'][s_wrt][-1].append((Pd[f_name][s_wrt].reshape((sd['num'], self.state_dict[s_wrt]['num']))))

                    # Store param Jac:
                    if self.param_bool == True:
                        for key_param in self.parameter_dict:
                            PJac = Pd[f_name][key_param].reshape(
                                (sd['num'], self.parameter_dict[key_param]['num']))

                            if not self.parameter_dict[key_param]['dynamic']:  # static parameter jac storage
                                if self.OStype == 'NS':
                                    partialtype = self.ode_system.partial_properties[f_name][key_param]['type']

                                    if partialtype == 'empty':
                                        if s_num == 0:
                                            sd['df_dp'][key_param].append(np.array([]))

                                    if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                                        if s_num == 0:
                                            sd['df_dp'][key_param].append(PJac)
                                        else:
                                            sd['df_dp'][key_param][-1] = sp.vstack((sd['df_dp'][key_param][-1], PJac))

                                    elif partialtype == 'std' or partialtype == 'cs_uc':
                                        if s_num == 0:
                                            sd['df_dp'][key_param].append(PJac)
                                        else:
                                            sd['df_dp'][key_param][-1] = np.vstack((sd['df_dp'][key_param][-1], PJac))

                                elif self.OStype == 'OM':
                                    
                                    if s_num == 0:
                                        sd['df_dp'][key_param].append(PJac)
                                    else:
                                        if sp.issparse(PJac):
                                            sd['df_dp'][key_param][-1] = sp.vstack((sd['df_dp'][key_param][-1], PJac))
                                        else:
                                            sd['df_dp'][key_param][-1] = np.vstack((sd['df_dp'][key_param][-1], PJac))

                            else:  # dynamic parameter jacobian storage
                                if self.OStype == 'NS':
                                    partialtype = self.ode_system.partial_properties[f_name][key_param]['type']

                                    if partialtype == 'empty':
                                        if s_num == 0:
                                            sd['df_dp'][key_param].append(np.array([]))
                                            sd['df_dp+'][key_param].append(np.array([]))

                                    elif partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                                        if s_num == 0:
                                            sd['df_dp'][key_param].append(PJac*self.GLM_C_minus[s_num][0])
                                            sd['df_dp+'][key_param].append(PJac*self.GLM_C[s_num][0])

                                        else:
                                            sd['df_dp'][key_param][-1] = sp.vstack((sd['df_dp'][key_param][-1], PJac*self.GLM_C_minus[s_num][0]))
                                            sd['df_dp+'][key_param][-1] = sp.vstack((sd['df_dp+'][key_param][-1], PJac*self.GLM_C[s_num][0]))

                                    elif partialtype == 'std' or partialtype == 'cs_uc':
                                        if s_num == 0:
                                            sd['df_dp'][key_param].append(PJac*self.GLM_C_minus[s_num][0])
                                            sd['df_dp+'][key_param].append(PJac*self.GLM_C[s_num][0])
                                        else:
                                            sd['df_dp'][key_param][-1] = np.vstack((sd['df_dp'][key_param][-1], PJac*self.GLM_C_minus[s_num][0]))
                                            sd['df_dp+'][key_param][-1] = np.vstack((sd['df_dp+'][key_param][-1], PJac*self.GLM_C[s_num][0]))

                                elif self.OStype == 'OM':
                                    if s_num == 0:
                                        sd['df_dp'][key_param].append(PJac*self.GLM_C_minus[s_num][0])
                                        sd['df_dp+'][key_param].append(PJac*self.GLM_C[s_num][0])
                                    else:
                                        if sp.issparse(PJac):
                                            sd['df_dp'][key_param][-1] = sp.vstack((sd['df_dp'][key_param][-1], PJac*self.GLM_C_minus[s_num][0]))
                                            sd['df_dp+'][key_param][-1] = sp.vstack((sd['df_dp+'][key_param][-1], PJac*self.GLM_C[s_num][0]))
                                        else:
                                            sd['df_dp'][key_param][-1] = np.vstack((sd['df_dp'][key_param][-1], PJac*self.GLM_C_minus[s_num][0]))
                                            sd['df_dp+'][key_param][-1] = np.vstack((sd['df_dp+'][key_param][-1], PJac*self.GLM_C[s_num][0]))
        #  Create vector of f's for next step
        for key in self.state_dict:
            sd = self.state_dict[key]
            sd['Yeval_current'] = np.concatenate(tuple(sd['k_temp'])).reshape(sd['shape_stage'])

    # part b: stage evaluation
    def evaluate_stage(self, t_now_index):
        # Compute function evaluation of stage. note Kronecker product.

        # Setting variables
        output_vals = []
        run_dict = {}
        for key in self.state_dict:
            sd = self.state_dict[key]

            output_vals.append(sd['f_name'])
            run_dict[key] = sd['Y_current'].reshape(sd['nn_shape'])

        # Computing F
        trm = time.time()
        P = self.ode_system.run_model(run_dict, output_vals)
        self.trm += (time.time() - trm)

        # Setting F
        for key in P:
            state_name = self.f2s_dict[key]
            self.state_dict[state_name]['Yeval_current'] = P[key].reshape(
                self.state_dict[state_name]['shape_stage'])
        return

    # part c: State Computation
    def compute_state(self, h):

        # Comuting state for each state
        for key in self.state_dict:
            sd = self.state_dict[key]
            self.state_dict[key]['y_current'] = h*(sd['B_kron'] * sd['Yeval_current']) + sd['V_kron']*sd['y_previous']
        return

    # COmpute JVP:
    def compute_JVP_phase(self, d_inputs_in, d_outputs_in, t_start_index=0, t_end_index=None, checkpoints=False):
        # Record time of JVP calculation
        start_JVP = time.time()
        if t_end_index == None:
            t_end_index = self.num_steps

        if self.display != None:
            print('Calculating Vector Jacobian Product ...')

        # Creating 'v' vector
        d_outputs = self.d_out
        for key in d_outputs_in:
            if key in self.profile_output_dict:
                self.profile_output_dict[key]['v'] = d_outputs_in[key].reshape(
                    self.profile_output_dict[key]['num'])
            elif key in self.field_output_dict:
                self.field_output_dict[key]['v'] = d_outputs_in[key].reshape(
                    self.field_output_dict[key]['num'])
            elif key in self.state_output_name_tuple:
                state_name = self.output_state_name_dict[key]['state_name']
                self.output_state_name_dict[key]['v'] = np.array(d_outputs_in[key]).reshape(
                    self.state_dict[state_name]['output_num'])
        # Initial Condition if applicable
        jvp_REV = {}
        for key in d_inputs_in:
            # Set current JVP as given inputs
            jvp_REV[key] = d_inputs_in[key].reshape((
                np.prod(d_inputs_in[key].shape),))

        # If first timestep, need to perform some extra operations for initial conditions
        if t_start_index == 0 and self.all_fixed_IC == False:

            # JVP for profile outputs IC
            if self.profile_outputs_bool == True:
                run_dict = {}
                outs = []
                state_and_params = []

                # Setting Key
                for key in self.profile_output_dict:
                    outs.append(key)

                # Setting initial states for profile output initial JVP
                for state_name in self.state_dict:
                    sd = self.state_dict[state_name]
                    state_and_params.append(state_name)
                    temp = np.empty(sd['nn_shape_profile'])
                    icname = self.state_dict[state_name]['IC_name']
                    si = self.IC_dict[icname]
                    temp[0] = si['val'].reshape(sd['shape'])
                    run_dict[state_name] = temp

                # Setting initial params for profile output initial JVP
                for param_name in self.parameter_dict:
                    if self.parameter_dict[param_name]['dynamic'] == True:
                        temp = np.array([self.parameter_dict[param_name]['val'][0].reshape(self.parameter_dict[param_name]['shape'])])
                        run_dict[param_name] = temp
                    state_and_params.append(param_name)

                # set and compute derivatives for initial time
                self.profile_outputs_system.set_vars(run_dict)
                self.profile_outputs_system.run_model({}, [])
                d = self.profile_outputs_system.compute_total_derivatives(
                    outs, state_and_params)

                # Setting initial JVP
                for key in self.profile_output_dict:
                    pd = self.profile_output_dict[key]
                    v_cur = self.profile_output_dict[key]['v'][0:pd['num_single']]

                    # profile JVP for initial condition
                    for state_name in self.state_dict:
                        if self.state_dict[state_name]['fixed_input'] == True:
                            continue
                        dnow = d[key][state_name].reshape(
                            (pd['num_single'], self.state_dict[state_name]['num']))
                        icname = self.state_dict[state_name]['IC_name']
                        if self.profile_outputs_system.system_type == 'NS':
                            ptype = self.profile_outputs_system.partial_properties[key][state_name]['type']
                            if ptype == 'empty':
                                continue
                            elif ptype == 'std' or ptype == 'cs_uc':
                                jvp_REV[icname] += v_cur.dot(dnow)
                            elif ptype == 'row_col' or ptype == 'row_col_val' or ptype == 'sparse':
                                jvp_REV[icname] += v_cur*(dnow)
                        if self.profile_outputs_system.system_type == 'OM':

                            if sp.issparse(dnow):
                                # print(state_name,jvp_REV[icname].shape, type(jvp_REV[icname]),  v_cur.shape, type(v_cur), dnow.shape, type(dnow))
                                # jvp_REV[icname] += v_cur*(dnow)
                                jvp_REV[icname] += v_cur@(dnow)

                                # print('fine', jvp_REV[icname].shape)
                            else:
                                jvp_REV[icname] += v_cur.dot(dnow)

                    # profile JVP for initial params
                    for param_name in self.parameter_dict:
                        param_d = self.parameter_dict[param_name]
                        if param_d['fixed_input'] == True:
                            continue
                        dnow = d[key][param_name].reshape(pd['num_single'], param_d['num'])
                        if param_d['dynamic'] == False:
                            if self.profile_outputs_system.system_type == 'NS':
                                ptype = self.profile_outputs_system.partial_properties[key][param_name]['type']
                                if ptype == 'empty':
                                    continue
                                elif ptype == 'std' or ptype == 'cs_uc':
                                    jvp_REV[param_name] += v_cur.dot(dnow)
                                elif ptype == 'row_col' or ptype == 'row_col_val' or ptype == 'sparse':
                                    jvp_REV[param_name] += v_cur*(dnow)
                            if self.profile_outputs_system.system_type == 'OM':
                                if sp.issparse(dnow):
                                    jvp_REV[param_name] += v_cur@(dnow)
                                else:
                                    jvp_REV[param_name] += v_cur.dot(dnow)
                                
                        else:
                            if self.profile_outputs_system.system_type == 'NS':
                                ptype = self.profile_outputs_system.partial_properties[key][param_name]['type']
                                if ptype == 'empty':
                                    continue
                                elif ptype == 'std' or ptype == 'cs_uc':
                                    jvp_REV[param_name][0:param_d['num']] += v_cur.dot(dnow)
                                elif ptype == 'row_col' or ptype == 'row_col_val' or ptype == 'sparse':
                                    jvp_REV[param_name][0:param_d['num']] += v_cur*(dnow)
                            if self.profile_outputs_system.system_type == 'OM':
                                if sp.issparse(dnow):
                                    jvp_REV[param_name][0:param_d['num']] += v_cur@(dnow)
                                else:
                                    jvp_REV[param_name][0:param_d['num']] += v_cur.dot(dnow)
            # Field outputs
            for key in self.field_output_dict:
                state_name = self.field_output_dict[key]['state_name']

                # If fixed initial condition, no need to compute JVP
                if self.state_dict[state_name]['fixed_input'] == True:
                    continue

                # Else, compute JVP
                icname = self.state_dict[state_name]['IC_name']
                si = self.IC_dict[icname]
                jvp_REV[icname] += self.field_output_dict[key]['v'] * \
                    self.field_output_dict[key]['coefficients'][0]

            # state_outputs:
            for key in self.state_output_name_tuple:
                state_name = self.output_state_name_dict[key]['state_name']

                # If fixed initial condition, no need to compute JVP
                if self.state_dict[state_name]['fixed_input'] == True:
                    continue
                # Else, compute JVP

                icname = self.state_dict[state_name]['IC_name']
                jvp_REV[icname] += self.output_state_name_dict[key]['v'][0: self.state_dict[state_name]['num']]

        # Time range
        time_range = range(t_start_index+1, t_end_index+1)
        # print('TIME RANGE:', list(time_range))

        # updating parameters
        run_dict_cur = {}
        for key in self.parameter_dict:
            if self.parameter_dict[key]['dynamic'] == False:
                run_dict_cur[key] = self.parameter_dict[key]['val']
        self.ode_system.set_vars(run_dict_cur)

        # Setting some things up
        # Part a:
        # wanted state derivatives list
        of_in = []
        for state_key in self.state_dict:
            sd = self.state_dict[state_key]
            of_in.append(state_key)

            # initializing dictionaries for jacobians
            sd['Y_prime_current_T'] = {}
            sd['df_dp_current'] = {}
            sd['df_dp_current+'] = {}

            if self.num_steps == t_end_index:
                self.state_dict[state_key]['psi_tA_prev'] = np.ones((sd['num_stage_state']))

        # part b:
        # part c:
        # cumulative sum:
        if 1 in time_range:
            S_mid = sp.csr_matrix(np.zeros(()))
        # Parameters dictionary
        wrt_param = []
        num_param = 0
        for key in self.parameter_dict:
            wrt_param.append(key)
            if self.parameter_dict[key]['dynamic']:
                num_param += self.parameter_dict[key]['num_dynamic']
            else:
                num_param += self.parameter_dict[key]['num']

        of_in_F = []
        # f dictionary
        for f_key in self.f2s_dict:
            of_in_F.append(f_key)

        # Backwards time loop ---------------------------------
        for t in reversed(time_range):
            ref = time.time()
            time_now_index = t-1  # Absolute time vector index

            # Index that goes from zero to length of timerange regardless of phase
            rel_index = t - t_start_index

            if self.times['type'] == 'step_vector':
                h = self.times['val'][time_now_index]

            # Do not continue iteration if time index is initial condition.
            # This is already taken care of before time loop
            # print(time_now_index, rel_index)
            if time_now_index + 1 == 0:
                continue

            # Setting Variables for profile outputs and pulling stored values
            run_dict_cur = {}
            profile_run_dict = {}
            profile_ofs = []
            profile_wrts = []

            # Set states
            for key in self.state_dict:
                sd = self.state_dict[key]
                # Precompute for Psi A
                sd['nhAkron'] = -h*sd['A_kron']
                if not self.explicit:
                    sd['nhAkronT'] = -h*sd['A_kronT']

                # Pulling correct inputs at right timestep
                sd['y_current'] = sd['y_storage'][:, rel_index]
                sd['Y_evalcurrent'] = sd['Yeval_full'][:, rel_index:rel_index+1]

                # Prepare to run ODE by setting input of states equal to stage Y
                if checkpoints == True:
                    # If implicit, we only run the ODE once across all stages vectorized.
                    Y_current = sd['A_kron']*h*sd['Y_evalcurrent'].reshape(sd['shape_stage']) + \
                        sd['U_kron']*(sd['y_storage'][:, rel_index-1])

                    # Set ODE states to stage value
                    if not self.explicit:
                        run_dict_cur[key] = Y_current.reshape(sd['nn_shape'])
                    else:
                        sd['Y_current_jvp'] = Y_current.reshape(sd['nn_shape'])

            # If checkpoint scheme, compute derivatives again:
            if checkpoints == True:

                # If implicit, we only run the ODE once across all stages vectorized.
                if not self.explicit:

                    # Set dynamic parameters
                    for key in self.parameter_dict:
                        if self.parameter_dict[key]['dynamic'] == True:
                            temp = self.parameter_dict[key]['val_nodal'][time_now_index].reshape(self.parameter_dict[key]['nn_shape'])
                            run_dict_cur[key] = temp
                    self.ode_system.set_vars(run_dict_cur)

                    # compute the derivatives
                    # RUN MODEL SHOULD NOT HAVE TO BE CALLED, NEED CSDL TO BE FIXED
                    self.ode_system.run_model({}, [])
                    partials_current = self.ode_system.compute_total_derivatives(
                        self.of_list, self.wrt_list)

                # If explicit, we run the ODE (num_stages) times.
                else:
                    # Loop through each stage
                    for s_num in range(self.num_stages):

                        # Set dynamic parameters:
                        for key in self.parameter_dict:
                            if self.parameter_dict[key]['dynamic'] == True:
                                run_dict_cur[key] = self.parameter_dict[key]['val_nodal'][time_now_index][s_num].reshape(self.parameter_dict[key]['nn_shape_exp'])

                        # Set states:
                        for key in self.state_dict:
                            sd = self.state_dict[key]
                            run_dict_cur[key] = sd['Y_current_jvp'][s_num].reshape(sd['nn_shape_exp'])

                        self.ode_system.set_vars(run_dict_cur)

                        # Compute derivatives
                        # RUN MODEL SHOULD NOT HAVE TO BE CALLED, NEED CSDL TO BE FIXED
                        self.ode_system.run_model({}, [])
                        partials_current = self.ode_system.compute_total_derivatives(
                            self.of_list, self.wrt_list)

                        # set current partials
                        for s_of in self.state_dict:
                            sd = self.state_dict[s_of]
                            f_name = sd['f_name']
                            for s_wrt in self.state_dict:
                                swrt = self.state_dict[s_wrt]

                                # Just like in the forward intergation, we store a list with each element corresponding to a stage
                                if s_num == 0:
                                    sd['Y_prime_current_T'][s_wrt] = []

                                # Store state jac. very similar to fwd stage computation for explicit
                                if self.OStype == 'OM':
                                    if sp.issparse(partials_current[f_name][s_wrt]):
                                        sd['Y_prime_current_T'][s_wrt].append((partials_current[f_name][s_wrt].reshape((sd['num'], swrt['num']))).copy())
                                    else:
                                        sd['Y_prime_current_T'][s_wrt].append((partials_current[f_name][s_wrt].reshape((sd['num'], swrt['num']))))
                                else:  # 'NS'
                                    partialtype = self.ode_system.partial_properties[f_name][s_wrt]['type']

                                    if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                                        sd['Y_prime_current_T'][s_wrt].append((partials_current[f_name][s_wrt].reshape((sd['num'], swrt['num']))).copy())
                                    else:
                                        sd['Y_prime_current_T'][s_wrt].append((partials_current[f_name][s_wrt].reshape((sd['num'], swrt['num']))))

                            # Store param jac. very similar to fwd stage computation for explicit
                            for key_param in self.parameter_dict:
                                pd = self.parameter_dict[key_param]
                                PJac = partials_current[f_name][key_param].reshape((sd['num'], pd['num']))

                                if not pd['dynamic']:
                                    if self.OStype == 'NS':
                                        partialtype = self.ode_system.partial_properties[f_name][key_param]['type']

                                        if partialtype == 'empty':
                                            continue

                                        if s_num == 0:
                                            sd['df_dp_current'][key_param] = PJac
                                            continue

                                        if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                                            sd['df_dp_current'][key_param] = sp.vstack((sd['df_dp_current'][key_param], PJac))

                                        elif partialtype == 'std' or partialtype == 'cs_uc':
                                            sd['df_dp_current'][key_param] = np.vstack((sd['df_dp_current'][key_param], PJac))

                                    elif self.OStype == 'OM':
                                        if s_num == 0:
                                            sd['df_dp_current'][key_param] = PJac
                                            continue

                                        if sp.issparse(PJac):
                                            sd['df_dp_current'][key_param] = sp.vstack((sd['df_dp_current'][key_param], PJac))
                                        else:
                                            sd['df_dp_current'][key_param] = np.vstack((sd['df_dp_current'][key_param], PJac))

                                else:  # Dynamic
                                    if self.OStype == 'NS':
                                        partialtype = self.ode_system.partial_properties[f_name][key_param]['type']

                                        if partialtype == 'empty':
                                            continue

                                        if s_num == 0:
                                            sd['df_dp_current'][key_param] = PJac*self.GLM_C_minus[s_num][0]
                                            sd['df_dp_current+'][key_param] = PJac*self.GLM_C[s_num][0]
                                            continue

                                        if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                                            sd['df_dp_current'][key_param] = sp.vstack((sd['df_dp_current'][key_param], PJac*self.GLM_C_minus[s_num][0]))
                                            sd['df_dp_current+'][key_param] = sp.vstack((sd['df_dp_current+'][key_param], PJac*self.GLM_C[s_num][0]))
                                        elif partialtype == 'std' or partialtype == 'cs_uc':
                                            sd['df_dp_current'][key_param] = np.vstack((sd['df_dp_current'][key_param], PJac*self.GLM_C_minus[s_num][0]))
                                            sd['df_dp_current+'][key_param] = np.vstack((sd['df_dp_current+'][key_param], PJac*self.GLM_C[s_num][0]))

                                    elif self.OStype == 'OM':
                                        if s_num == 0:
                                            sd['df_dp_current'][key_param] = PJac*self.GLM_C_minus[s_num][0]
                                            sd['df_dp_current+'][key_param] = PJac*self.GLM_C[s_num][0]
                                            continue

                                        if sp.issparse(PJac):
                                            sd['df_dp_current'][key_param] = sp.vstack((sd['df_dp_current'][key_param], PJac*self.GLM_C_minus[s_num][0]))
                                            sd['df_dp_current+'][key_param] = sp.vstack((sd['df_dp_current+'][key_param], PJac*self.GLM_C[s_num][0]))
                                        else:
                                            sd['df_dp_current'][key_param] = np.vstack((sd['df_dp_current'][key_param], PJac*self.GLM_C_minus[s_num][0]))
                                            sd['df_dp_current+'][key_param] = np.vstack((sd['df_dp_current+'][key_param], PJac*self.GLM_C[s_num][0]))
            for key in self.state_dict:
                sd = self.state_dict[key]

                # If not checkpointing scheme, pull derivatives from stored tensor
                if checkpoints == False:
                    if not self.explicit:
                        for s_wrt in self.state_dict:
                            swrt = self.state_dict[s_wrt]
                            sd['Y_prime_current_T'][s_wrt] = sd['Y_prime_full'][s_wrt][rel_index].reshape(
                                (sd['num_stage_state'], swrt['num_stage_state'])).transpose()

                    else:  # explicit
                        for s_wrt in self.state_dict:
                            swrt = self.state_dict[s_wrt]
                            sd['Y_prime_current_T'][s_wrt] = []
                            for stage_jac in sd['Y_prime_full'][s_wrt][rel_index]:
                                sd['Y_prime_current_T'][s_wrt].append(stage_jac)
                # If checkpoint scheme, set derivatives from recomputed derivatives.
                else:  # checkpointing
                    if not self.explicit:  # we already set derivatives for explicit case
                        f_name = sd['f_name']
                        for s_wrt in self.state_dict:
                            swrt = self.state_dict[s_wrt]

                            sd['Y_prime_current_T'][s_wrt] = partials_current[f_name][s_wrt].reshape(
                                (sd['num_stage_state'], swrt['num_stage_state'])).transpose()

                temp = np.empty(sd['nn_shape_profile'])
                temp[0] = sd['y_current'].reshape(sd['shape'])

                # setting dictionary for set_vars profile
                if self.profile_outputs_bool == True:

                    profile_wrts.append(key)
                    profile_run_dict[key] = temp
                # # setting dictionary for set_vars ode
                # if self.profile_outputs_bool == True:

                #     for profiles in self.profile_output_dict:
                #         profile_ofs.append(profiles)
                #         if self.profile_output_dict[profiles]['state_name'] == key:
                #             # Creating array for input
                #             temp = np.empty(sd['nn_shape_profile'])
                #             temp[0] = self.state_dict[key]['y_current'].reshape(
                #                 sd['shape'])

                #             # setting dictionary for set_vars profile
                #             profile_wrts.append(key)
                #             profile_run_dict[key] = temp
            if self.profile_outputs_bool == True:

                # set dynamic parameters:
                # CHANGE BACK:
                for key in self.parameter_dict:
                    if self.parameter_dict[key]['dynamic'] == True:
                        profile_run_dict[key] = self.parameter_dict[key]['val'][time_now_index+1].reshape(self.parameter_dict[key]['shape'])

                # run and compute totals of profile outputs
                profile_wrts.extend(self.parameter_dict.keys())
                for profiles in self.profile_output_dict:
                    profile_ofs.append(profiles)
                self.profile_outputs_system.set_vars(profile_run_dict)
                self.profile_outputs_system.run_model({}, [])
                P = self.profile_outputs_system.compute_total_derivatives(
                    profile_ofs, profile_wrts)

                # profile JVP for initial params
                for profile_output in self.profile_output_dict:
                    pd = self.profile_output_dict[profile_output]
                    v_cur = pd['v'][pd['num_single'] * (t):pd['num_single']*(t+1)]

                    for param_name in self.parameter_dict:
                        param_d = self.parameter_dict[param_name]
                        if param_d['fixed_input'] == True:
                            continue
                        dnow = P[profile_output][param_name].reshape(pd['num_single'], param_d['num'])
                        if param_d['dynamic'] == False:
                            if self.profile_outputs_system.system_type == 'NS':
                                ptype = self.profile_outputs_system.partial_properties[profile_output][param_name]['type']
                                if ptype == 'empty':
                                    continue
                                elif ptype == 'std' or ptype == 'cs_uc':
                                    jvp_REV[param_name] += v_cur.dot(dnow)
                                elif ptype == 'row_col' or ptype == 'row_col_val' or ptype == 'sparse':
                                    jvp_REV[param_name] += v_cur*(dnow)
                            if self.profile_outputs_system.system_type == 'OM':
                                if sp.issparse(dnow):
                                    jvp_REV[param_name] += v_cur@(dnow)
                                else:
                                    jvp_REV[param_name] += v_cur.dot(dnow)
                        else:
                            if self.profile_outputs_system.system_type == 'NS':
                                ptype = self.profile_outputs_system.partial_properties[profile_output][param_name]['type']
                                if ptype == 'empty':
                                    continue
                                elif ptype == 'std' or ptype == 'cs_uc':
                                    jvp_REV[param_name][(t)*param_d['num']:(t+1) * param_d['num']] += v_cur.dot(dnow)
                                elif ptype == 'row_col' or ptype == 'row_col_val' or ptype == 'sparse':
                                    jvp_REV[param_name][(t)*param_d['num']:(t+1) * param_d['num']] += v_cur*(dnow)
                            if self.profile_outputs_system.system_type == 'OM':
                                if sp.issparse(dnow):
                                    jvp_REV[param_name][(t)*param_d['num']:(t+1) * param_d['num']] += v_cur@(dnow)
                                else:
                                    jvp_REV[param_name][(t)*param_d['num']:(t+1) * param_d['num']] += v_cur.dot(dnow)
            else:
                P = None

            #  ----------------------------------------------------- PART C / PART B -----------------------------------------------------
            # ===UNCOMMENT FOR PROFILING===
            # from line_profiler import LineProfiler
            # lp = LineProfiler()
            # lp_wrapper = lp(self.compute_JVP_psi_cb)
            # lp_wrapper(P, t, time_now_index, h)
            # lp.print_stats()
            # ===UNCOMMENT FOR PROFILING===
            self.compute_JVP_psi_cb(P, t, time_now_index, h)

            #  ----------------------------------------------------- PART A -----------------------------------------------------

            # EXPLICIT SOLVER:
            if self.explicit:
                # ===UNCOMMENT FOR PROFILING===
                # from line_profiler import LineProfiler
                # lp = LineProfiler()
                # lp_wrapper = lp(self.compute_JVP_psi_a_explicit)
                # lp_wrapper(h)
                # lp.print_stats()
                # ===UNCOMMENT FOR PROFILING===
                self.compute_JVP_psi_a_explicit(h)

            # DIRECT SOLVER (FOR IMPLICIT):
            elif self.implicit_solver_jvp == 'direct':
                # ===UNCOMMENT FOR PROFILING===
                # from line_profiler import LineProfiler
                # lp = LineProfiler()
                # lp_wrapper = lp(self.compute_JVP_psi_a)
                # lp_wrapper(h)
                # lp.print_stats()
                # ===UNCOMMENT FOR PROFILING===
                self.compute_JVP_psi_a(h)

            # BGS SOLVER (FOR IMPLICIT):
            elif self.implicit_solver_jvp == 'iterative':
                # ===UNCOMMENT FOR PROFILING===
                # from line_profiler import LineProfiler
                # lp = LineProfiler()
                # lp_wrapper = lp(self.compute_JVP_psi_a_BGS)
                # lp_wrapper(h)
                # lp.print_stats()
                # ===UNCOMMENT FOR PROFILING===
                self.compute_JVP_psi_a_BGS(h)

            # ---------------------------------------------- CUMULATIVE SUMMATION -----------------------------------------------
            # Initial Condition S:
            if t == 1:
                for ic_name in self.IC_dict:
                    # No need to compte JVP if fixed initial conditions
                    if self.IC_dict[ic_name]['fixed_input'] == True:
                        continue

                    # Else, compute JVP
                    state_name = self.IC_dict[ic_name]['state_name']
                    sd = self.state_dict[state_name]
                    jvp_REV[ic_name] += -sd['psi_tA'] * sd['U_kron']-sd['psi_tC']*sd['V_kron']

            # Parameters S:
            if self.param_bool == True and self.all_fixed_parameters == False:
                for state in self.state_dict:
                    sd = self.state_dict[state]
                    f_name = sd['f_name']
                    for key_param in self.parameter_dict:
                        pd = self.parameter_dict[key_param]
                        # Need to reshape if dynamic parameters are dynamic
                        # If checkpointing, pull derivative and apply transform

                        empty_jac = False
                        if self.OStype == 'NS':
                            if self.ode_system.partial_properties[f_name][key_param]['type'] == 'empty':
                                empty_jac = True

                        if not empty_jac:
                            if checkpoints == True:
                                if not self.explicit:
                                    if pd['dynamic'] == False:
                                        sd['df_dp_current'][key_param] = (
                                            partials_current[f_name][key_param])

                                    else:
                                        df_dp = partials_current[f_name][key_param].reshape(
                                            (sd['num_stage_state'], self.num_stages*pd['num']))

                                        if sp.issparse(df_dp):
                                            df_dp_store = df_dp@pd['stage2state_transform_s']
                                            df_dp_next = df_dp@pd['stage2state_transform_s+']
                                        else:
                                            df_dp_store = df_dp.dot(pd['stage2state_transform_d'])
                                            df_dp_next = df_dp.dot(pd['stage2state_transform_d+'])
                                        sd['df_dp_current'][key_param] = (df_dp_store)
                                        sd['df_dp_current+'][key_param] = (df_dp_next)
                            else:  # Not checkpointing
                                # apply transform
                                sd['df_dp_current'][key_param] = sd['df_dp'][key_param][rel_index].reshape(
                                    (sd['num_stage_state'], pd['num']))
                                if pd['dynamic']:
                                    sd['df_dp_current+'][key_param] = sd['df_dp+'][key_param][rel_index].reshape(
                                        (sd['num_stage_state'], pd['num']))

                for s_key in self.state_dict:
                    sd = self.state_dict[s_key]
                    f_name = sd['f_name']

                    # This precomputation SHOULD be faster but it isnt...
                    # psiA_AB = (sd['psi_tA']*sd['nhAkron'] - sd['psi_tB'])
                    for p_key in sd['df_dp_current']:

                        # No need to compute JVP if fixed parameters
                        if self.parameter_dict[p_key]['fixed_input'] == True:
                            continue

                        # Else, compute JVP
                        pd = self.parameter_dict[p_key]

                        if pd['dynamic']:
                            t_plus_vec = [0, 1]
                        else:
                            t_plus_vec = [0]

                        for t_plus in t_plus_vec:
                            if t_plus == 1:
                                sd['df_dp_current'][p_key] = sd['df_dp_current+'][p_key]
                            # Native System
                            if self.OStype == 'NS':
                                ptype = self.ode_system.partial_properties[f_name][p_key]['type']
                                if ptype == 'empty':
                                    continue

                                # Getting dfdp:
                                dfdp_c = sd['df_dp_current'][p_key].reshape((sd['num_stage_state'], pd['num']))

                                # Calculating JVP:
                                if ptype == 'row_col' or ptype == 'row_col_val' or ptype == 'sparse':
                                    # jvp_REVadd = -h * sd['psi_tA']*(sd['A_kron'] * dfdp_c) - sd['psi_tB']*dfdp_c
                                    jvp_REVadd = (sd['psi_tA']*sd['nhAkron'] - sd['psi_tB']) * dfdp_c

                                    # This precomputation SHOULD be faster but it isnt...
                                    # jvp_REVadd = psiA_AB * dfdp_c

                                elif ptype == 'std' or ptype == 'cs_uc':
                                    jvp_REVadd = sd['psi_tA'].dot(sd['nhAkron'].dot(dfdp_c)) - sd['psi_tB'].dot(dfdp_c)

                                    # This precomputation SHOULD be faster but it isnt...
                                    # jvp_REVadd = psiA_AB.dot(dfdp_c)

                            # OpenMDAO System
                            elif self.OStype == 'OM':
                                # Getting dfdp:
                                dfdp_c = sd['df_dp_current'][p_key].reshape((sd['num_stage_state'], pd['num']))
                                if sp.issparse(dfdp_c):
                                    jvp_REVadd = (sd['psi_tA']*sd['nhAkron'] - sd['psi_tB']) @ dfdp_c
                                else:
                                    jvp_REVadd = sd['psi_tA'].dot(sd['nhAkron'].dot(dfdp_c)) - sd['psi_tB'].dot(dfdp_c)

                            if pd['dynamic'] == False:
                                jvp_REV[p_key] += jvp_REVadd
                            else:
                                jvp_REV[p_key][(t-1+t_plus)*pd['num']:(t+t_plus) * pd['num']] += jvp_REVadd
                                # if t-1+t_plus == self.num_steps:
                                #     jvp_REV[p_key][(t-1)*pd['num']:(t) * pd['num']] += jvp_REVadd
                                # else:
                                #     jvp_REV[p_key][(t-1+t_plus)*pd['num']:(t+t_plus) * pd['num']] += jvp_REVadd

            # h S:
            if self.times['fixed_input'] == False:
                for key in self.state_dict:
                    sd = self.state_dict[key]
                    jvp_REV[self.times['name']][t-1] += -sd['psi_tA'].dot(sd['A_kron']*sd['Y_evalcurrent']) - sd['psi_tC'].dot(sd['B_kron']*sd['Y_evalcurrent'])

            # Updating psi iterations:
            for key in self.state_dict:

                # set Jacobians to zero.
                for s_wrt in self.state_dict:
                    self.state_dict[key]['Y_prime_current_T'][s_wrt] = 0

                # Set Psi's for next iteration.
                self.state_dict[key]['psi_tA_prev'] = self.state_dict[key]['psi_tA']
                self.state_dict[key]['psi_tC_prev'] = self.state_dict[key]['psi_tC']

        d_inputs_return = {}
        for key in d_inputs_in:
            d_inputs_return[key] = jvp_REV[key]
        end_JVP = time.time()
        if self.display != None:
            print('JVP time:', (end_JVP - start_JVP))
        return d_inputs_return

    def compute_JVP_psi_cb(self, P, t, time_now_index, h):
        # Creating psitilda_C for each state:
        for key in self.state_dict:

            # Initializing 'Psi's'
            sd = self.state_dict[key]

            self.state_dict[key]['psi_tC'] = np.zeros((sd['num']))
            self.state_dict[key]['psi_tA'] = np.zeros(
                (sd['num_stage_state']))

            # PART A  -----------:
            for profile_key in self.profile_output_dict:
                pd = self.profile_output_dict[profile_key]
                dpds_temp = (P[profile_key][key].T)

                # OLD: was reshaping instead of transposing
                # dpds_temp = (P[profile_key][key].reshape((sd['num'], pd['num_single'])))


                v_current = pd['v'][pd['num_single'] * (t):pd['num_single']*(t+1)]

                if self.profile_outputs_system.system_type == 'NS':
                    ptype = self.profile_outputs_system.partial_properties[
                        profile_key][key]['type']

                    if ptype == 'empty':
                        continue

                    if ptype == 'std' or ptype == 'cs_uc':
                        self.state_dict[key]['psi_tC'] += - \
                            dpds_temp.dot(v_current)
                    elif ptype == 'row_col' or ptype == 'row_col_val' or ptype == 'sparse':
                        self.state_dict[key]['psi_tC'] += - \
                            dpds_temp*(v_current)
                elif self.profile_outputs_system.system_type == 'OM':
                    
                    if sp.issparse(dpds_temp):
                        self.state_dict[key]['psi_tC'] += - \
                            dpds_temp@(v_current)
                    else:
                        self.state_dict[key]['psi_tC'] += - \
                            dpds_temp.dot(v_current)

            for field_key in self.field_output_dict:
                if self.field_output_dict[field_key]['state_name'] == key:
                    v_current = self.field_output_dict[field_key]['v']
                    self.state_dict[key]['psi_tC'] += -(
                        self.field_output_dict[field_key]['coefficients'][t])*(v_current)

            if sd['output_bool'] == True:
                osnd = self.output_state_name_dict[sd['output_name']]
                v_current = osnd['v'][sd['num'] * (t):sd['num']*(t+1)]
                self.state_dict[key]['psi_tC'] += -v_current

            if time_now_index+1 != self.num_steps:
                self.state_dict[key]['psi_tC'] += (sd['U_kronT']*sd['psi_tA_prev'] + sd['V_kronT']*sd['psi_tC_prev'])

            # PART B ----------- :
            self.state_dict[key]['psi_tB'] = sd['B_kronT']*(h*self.state_dict[key]['psi_tC'])

    def compute_JVP_psi_a(self, h):

        # Explicit calculation of third step's psi_tilda
        A_rhs_full = np.zeros(self.all_states_num*self.num_stages)
        for s_of in self.state_dict:
            sd = self.state_dict[s_of]
            self.state_dict[s_of]['psi_tA_iter'] = sd['psi_tA_prev']

            # Create Columns for dF(current state)/dy(all states)
            for s_wrt in self.state_dict:
                swrt = self.state_dict[s_wrt]
                # print(s_of,s_wrt)
                if self.OStype == 'NS':
                    f_name = sd['f_name']
                    ptype = self.ode_system.partial_properties[f_name][s_wrt]['type']
                    if ptype == 'empty':
                        continue
                    # print((sd['Y_prime_full'][s_wrt]))
                    pd = sd['Y_prime_current_T'][s_wrt]
                    # pd = sd['Y_prime_full'][s_wrt][rel_index].reshape(
                    #     (sd['num_stage_state'], swrt['num_stage_state']))
                    if ptype == 'row_col_val' or ptype == 'row_col' or ptype == 'sparse':
                        A_rhs_full[swrt['stage_ind'][0]:swrt['stage_ind']
                                   [1]] += pd*sd['psi_tB']
                    elif ptype == 'std' or ptype == 'cs_uc':
                        A_rhs_full[swrt['stage_ind'][0]:swrt['stage_ind']
                                   [1]] += pd.dot(sd['psi_tB'])
                elif self.OStype == 'OM':
                    # print(len(sd['Y_prime_full'][s_wrt]))
                    # print(time_now_index)
                    pd = sd['Y_prime_current_T'][s_wrt]

                    if sp.issparse(pd):
                        A_rhs_full[swrt['stage_ind'][0]:swrt['stage_ind']
                                   [1]] += pd@sd['psi_tB']
                    else:
                        A_rhs_full[swrt['stage_ind'][0]:swrt['stage_ind']
                                [1]] += (pd).dot(sd['psi_tB'])

        # Create block matrix for jacobians NEEDS TO CHANGE?
        # Block matrix for jacobian
        bmat_list = []

        for state_wrt in self.state_dict:
            sw = self.state_dict[state_wrt]
            rows = []
            for state_of in self.state_dict:
                so = self.state_dict[state_of]
                # I-ah if derivative of itself
                if self.OStype == 'NS':
                    f_name = so['f_name']
                    ptype = self.ode_system.partial_properties[f_name][state_wrt]['type']

                    if ptype == 'empty':
                        entry = sp.csc_matrix(
                            (sw['num_stage_state'], so['num_stage_state']))

                    pd = so['Y_prime_current_T'][state_wrt]

                    if ptype == 'row_col_val' or ptype == 'row_col' or ptype == 'sparse':
                        if state_wrt == state_of:
                            entry = so['full_eye'] + pd*so['nhAkronT']
                        else:
                            entry = pd*so['nhAkronT']

                    elif ptype == 'std' or ptype == 'cs_uc':
                        pd = sp.csc_matrix(pd)
                        if state_wrt == state_of:
                            entry = so['full_eye'] + pd*so['nhAkronT']
                        else:
                            entry = pd*so['nhAkronT']
                # OPENMDAO
                else:
                    pd = so['Y_prime_current_T'][state_wrt]
                    pd = sp.csc_matrix(pd)
                    if state_wrt == state_of:
                        entry = so['full_eye'] + pd*so['nhAkronT']
                    else:
                        entry = pd*so['nhAkronT']

                rows.append(entry)
            bmat_list.append(rows)

        LHS = sp.bmat(bmat_list, format='csc')

        psi_ta = spln.spsolve(LHS, A_rhs_full)

        for states in self.state_dict:
            sd = self.state_dict[states]
            self.state_dict[states]['psi_tA'] = psi_ta[sd['stage_ind']
                                                       [0]:sd['stage_ind'][1]]

    def compute_JVP_psi_a_BGS(self, h):

        FPI_iteration_num = 0

        # Before BGS iteration:
        for s_wrt in self.state_dict:
            sw = self.state_dict[s_wrt]
            # Precompute constants that do not change with A
            sw['psi_tA_iter'] = sw['psi_tA_prev']
            sw['hA_kronT'] = h*sw['A_kronT']
            sw['const_tb'] = {}
            for s_of in self.state_dict:
                so = self.state_dict[s_of]

                if self.OStype == 'NS':

                    f_name = so['f_name']
                    partialtype = self.ode_system.partial_properties[f_name][s_wrt]['type']

                    if partialtype == 'empty':
                        sw['const_tb'][s_of] = np.zeros(sw['psi_tA'].shape)

                    if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                        sw['const_tb'][s_of] = so['Y_prime_current_T'][s_wrt]*so['psi_tB']

                    elif partialtype == 'std' or partialtype == 'cs_uc':
                        sw['const_tb'][s_of] = so['Y_prime_current_T'][s_wrt].dot(so['psi_tB'])

                elif self.OStype == 'OM':
                    if sp.issparse(so['Y_prime_current_T'][s_wrt]):
                        sw['const_tb'][s_of] = so['Y_prime_current_T'][s_wrt]@so['psi_tB']
                    else:
                        sw['const_tb'][s_of] = so['Y_prime_current_T'][s_wrt].dot(so['psi_tB'])

        # START BGS ITERATION:
        converged = False
        while not converged:

            # Precompute:
            for state in self.state_dict:
                sd = self.state_dict[state]
                sd['mult_r'] = sd['hA_kronT']*sd['psi_tA_iter']

            converged = True
            # Main operation of BGS:
            for s_wrt in self.state_dict:
                sw = self.state_dict[s_wrt]
                sw['psi_tA'] = np.zeros(sw['psi_tA_iter'].shape)
                for s_of in self.state_dict:
                    so = self.state_dict[s_of]

                    # Constant term
                    sw['psi_tA'] += sw['const_tb'][s_of]

                    # Iterative term:
                    if self.OStype == 'NS':
                        f_name = so['f_name']
                        partialtype = self.ode_system.partial_properties[f_name][s_wrt]['type']

                        if partialtype == 'empty':
                            continue

                        if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                            sw['psi_tA'] += so['Y_prime_current_T'][s_wrt]*so['mult_r']

                        elif partialtype == 'std' or partialtype == 'cs_uc':
                            sw['psi_tA'] += so['Y_prime_current_T'][s_wrt].dot(so['mult_r'])

                    elif self.OStype == 'OM':
                        if sp.issparse(so['Y_prime_current_T'][s_wrt]):
                            sw['psi_tA'] += so['Y_prime_current_T'][s_wrt]@so['mult_r']
                        else:
                            sw['psi_tA'] += so['Y_prime_current_T'][s_wrt].dot(so['mult_r'])

            # Check for convergence. If even one of the states are not converged, move to next iteration
            for state in self.state_dict:
                sd = self.state_dict[state]
                error_psiA = np.linalg.norm(sd['psi_tA'] - sd['psi_tA_iter'])
                # print(FPI_iteration_num, state, error_psiA)

                if error_psiA > 0.00000000001:
                    # Do not calculate next state error if current state is already not converged
                    converged = False
                    break

            # Set new iteration
            for state in self.state_dict:
                sd = self.state_dict[state]
                sd['psi_tA_iter'] = sd['psi_tA']

            # Keep track of iterations
            FPI_iteration_num += 1
            if FPI_iteration_num > 10:
                warnings.warn("iterative taking time to converge. An alternative is to use direct method: implicit_solver_jvp = 'direct'")
        # print('BGS ITER: ', FPI_iteration_num)

    def compute_JVP_psi_a_explicit(self, h):
        # Precompute h*A:
        for i in range(len(self.A_rows)):
            if i > 0:
                self.Ah[i] = h*(self.A_rows[i])

        # Preallocate psitemp to store and combine later
        # Split psi_b into stages
        for key in self.state_dict:
            sd = self.state_dict[key]
            sd['psi_A_temp'] = self.explicit_tools['psi_A_temp'].copy()

            sd['psi_tB_temp'] = []
            for inds in sd['psi_indices']:
                sd['psi_tB_temp'].append(sd['psi_tB'][inds[0]:inds[1]])

        for s_num in self.explicit_tools['rev_stage_iter']:
            for s_wrt in self.state_dict:
                sw = self.state_dict[s_wrt]
                # Initialize
                psi_current = np.zeros(sw['num'])
                for s_of in self.state_dict:
                    so = self.state_dict[s_of]

                    # Sum through stages
                    psi_temp = np.array(so['psi_tB_temp'][s_num])
                    for k in self.explicit_tools['stage_index_list'][s_num]:
                        psi_temp += (self.Ah[k][s_num]*so['psi_A_temp'][k])
                    # print(type(psi_current))

                    # Multiply with jacobian
                    if self.OStype == 'NS':
                        f_name = so['f_name']
                        partialtype = self.ode_system.partial_properties[f_name][s_wrt]['type']

                        if partialtype == 'empty':
                            continue

                        if partialtype == 'row_col' or partialtype == 'row_col_val' or partialtype == 'sparse':
                            psi_current += psi_temp*(so['Y_prime_current_T'][s_wrt][s_num])
                            # print('s ', s_of, s_wrt, psi_temp, psi_current, so['Y_prime_current_T'][s_wrt][s_num])

                        elif partialtype == 'std' or partialtype == 'cs_uc':
                            # ------------Alternative method? should be slower but doesnt seem so with scalars atleast ------------
                            # psi_current += (so['Y_prime_current_T'][s_wrt][s_num]).dot(so['psi_tB_temp'][s_num])
                            # for k in self.explicit_tools['stage_index_list'][s_num]:
                            #     psi_current += ((so['Y_prime_current_T'][s_wrt][s_num]).dot((self.Ah[k][s_num])*so['psi_A_temp'][k]))
                            # ------------Alternative method------------------------------------------------------------------------

                            # Compute psi_current. With numpy, A_transpose.dot(x) = x.dot(A)
                            # By using the RHS, we do not have to tranpose the jacobian which is much cheaper!
                            # psi_current += (so['Y_prime_current_T'][s_wrt][s_num]).dot(psi_temp)
                            psi_current += psi_temp.dot(so['Y_prime_current_T'][s_wrt][s_num])
                            # print('d ', s_of, s_wrt, psi_temp, psi_current, so['Y_prime_current_T'][s_wrt][s_num])

                    elif self.OStype == 'OM':
                        # print(psi_current, psi_temp.dot(so['Y_prime_current_T'][s_wrt][s_num]))
                        # print()
                        # print(type((so['Y_prime_current_T'][s_wrt][s_num])), s_of, s_wrt, s_num)
                        # print(type(psi_temp))
                        # print(type(psi_current))
                        # print(((so['Y_prime_current_T'][s_wrt][s_num])).shape, s_of, s_wrt, s_num)
                        # print((psi_temp).shape)
                        # print((psi_current).shape)

                        if sp.issparse(so['Y_prime_current_T'][s_wrt][s_num]):
                            psi_current += psi_temp@(so['Y_prime_current_T'][s_wrt][s_num])
                        else:
                            psi_current += psi_temp.dot(so['Y_prime_current_T'][s_wrt][s_num])

                # Store psi_A
                sw['psi_A_temp'][s_num] = psi_current

        for s_wrt in self.state_dict:
            sw = self.state_dict[s_wrt]
            # Create the actual vector
            sw['psi_tA'] = np.concatenate(tuple(sw['psi_A_temp']))

    def get_solver_model(self):
        # Creating explicit component for optimization loop
        # If approach = explicit component, return a group
        component = ODEModelTM()
        component.add_ODEProb(self)

        return component
