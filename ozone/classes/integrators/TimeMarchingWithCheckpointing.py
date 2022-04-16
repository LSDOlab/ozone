from ozone.classes.integrators.TimeMarching import TimeMarching
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.linalg as ln
import numpy as np
import matplotlib.pyplot as plt


class TimeMarchingWithCheckpointing(TimeMarching):
    """
    TimeMarchingWithCheckpointing is a child class of the TimeMarching class.
    The TimeMarchingWithCheckpointing class performs all calculations for the time-marching integration approach with checkpointing.
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

        # TimeMarching's post_setup_init is run first. Everything after this line is specific only to time-marching checkpointing
        # Create checkpoints
        self.checkpoints = []

        checkpoint_snapshot = {}
        for state in self.state_dict:
            checkpoint_snapshot[state] = {}

        # If number of checkpoints aren't given, automatically set number of checkpoints = sqrt(num timesteps)
        if self.num_checkpoints == None:
            self.num_checkpoints = round(self.num_steps**0.5)

        # Create checkpoint indices given user-defined number of checkpoints
        # Checkpoints are uniformly distributed accross time interval
        checkpoint_indices_reversed = [0]
        d_checks = (self.num_steps)/(self.num_checkpoints+1)
        for i in range(self.num_checkpoints):
            checkpoint_indices_reversed.append(int((i+1)*d_checks))
        self.checkpoint_indices = list(reversed(checkpoint_indices_reversed))

        # self.checkpoint_indices = [round(self.num_steps/2), 0]
        self.num_checkpoints = len(self.checkpoint_indices)
        for i in range(len(self.checkpoint_indices)):

            # With checkpointing, we can't plot states in real time or after integration.

            if i == 0:
                index_plus = self.num_steps+1
            else:
                index_plus = self.checkpoint_indices[i-1]
            index_minus = self.checkpoint_indices[i]
            checkpoint_preallocation_dict = {
                'index+': index_plus,
                'index-': index_minus,
                'checkpoint_snapshot-': checkpoint_snapshot.copy()}
            self.checkpoints.append(checkpoint_preallocation_dict)
        # print(' =====CHECKPOINTS: ', self.checkpoints)
        self.visualization = None

        return super().post_setup_init()

    def compute_JVP(self, d_inputs_in, d_outputs_in):
        """
        Compute Jacobian Vector product of the ODE from the last timestep to the first timestep.

        Parameters
        ----------
            d_inputs_in:

            d_outputs_in: 
        """

        # return self.compute_JVP_phase(d_inputs_in, d_outputs_in)

        integration_settings = {
            # Integration type = REV only, Do not calculate outputs and store states
            'integration_type': 'REV',
            'state_IC': self.IC_dict,
            't_index_end': None,
            't_index_start': None}

        din = d_inputs_in
        do = d_outputs_in
        # print('FIRST Din/Do', din, do)

        # Loop going over checkpoints from last point to first point
        for i in range(len(self.checkpoints)):
            # Compute integration from checkpoint i-1 to i
            integration_settings['state_IC'] = self.checkpoints[i]['checkpoint_snapshot-']
            if i > 0:
                integration_settings['t_index_end'] = self.checkpoints[i]['index+']+1
            else:
                integration_settings['t_index_end'] = self.checkpoints[i]['index+']
            integration_settings['t_index_start'] = self.checkpoints[i]['index-']
            self.integrate_ODE_phase(integration_settings)

            # Compute JVP from checkpoint i to i-1
            if i == len(self.checkpoints) - 1:
                t_start_i = 0
            else:
                t_start_i = self.checkpoints[i]['index-']

            if i == 0:
                t_end_i = self.checkpoints[i]['index+']-1
            else:
                t_end_i = self.checkpoints[i]['index+']
            din_n = self.compute_JVP_phase(
                din, do, t_start_index=t_start_i, t_end_index=t_end_i, checkpoints=True)
            din = din_n
            # Set new JVP vector
            # print(self.checkpoints[i])
        return din

    def integrate_ODE(self):
        """
        Integrates the ODE from the first timestep to the last timestep.
        """

        integration_settings = {
            # Integration type = FWD only, fwd integration only calculates outputs and checkpoints
            'integration_type': 'FWD',
            'state_IC': self.IC_dict,
            't_index_end': None,
            't_index_start': None}

        self.integrate_ODE_phase(integration_settings)
        # print(' =====CHECKPOINTS: ', self.checkpoints)

    def setup_integration(self):
        """
        Method that gets called before each integration (for checkpointing)
        """
        # Creating time vector
        self.time_vector_full = np.zeros((self.num_steps+1))
        if self.times['type'] == 'step_vector':
            self.time_vector_full[1:self.num_steps +
                                  1] = np.cumsum(self.times['val'])
            self.h_vector_full = self.times['val']
