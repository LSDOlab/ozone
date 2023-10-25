from ozone.classes.glm_matrices.GLMs import get_integration_method
import matplotlib.pyplot as plt
import numpy as np
# import openmdao.api as om
import time
import scipy.sparse as sp


class IntegratorBase(object):
    '''
    IntegratorBase is the parent class of the time-marching, solver-based and optimizer-based integration approaches.
    Therefore, the computations done in this class are needed for all three approaches.
    '''

    def __init__(self,
                 method,
                 num_times=1,
                 display='default',
                 error_tolerance=0.000000001,
                 visualization=None,
                 num_checkpoints=None,
                 implicit_solver_jvp='direct',
                 implicit_solver_fwd='direct'):
        """
        Arguments should be called from ODEProblem class.

        Parameters
        ----------
            method: str
                Numerical method to use for time integration
            num_steps: int
                Number of timesteps for numerical integration
            display: str
                Settings for what text to display during/after the integration process
            error_tolerance: float
                Error tolerance for implicit stage calculation nonlinear solver
            visualization: float
                Settings for what plot to display during/after the integration process
            num_checkpoints:
                Number of checkpoints if time-marching with checkpoints are used. Not used if approach is time-marching, solver-based
                or collocation
        """

        # Set some settings for all integration methods: Plotting type, display type, number of timesteps, error tolerance for stage calculations
        self.visualization = visualization
        self.display = display
        self.num_times = num_times
        self.num_steps = num_times-1
        self.error_tolerance = error_tolerance
        self.method = method
        self.num_checkpoints = num_checkpoints
        self.implicit_solver_jvp = implicit_solver_jvp
        self.implicit_solver_fwd = implicit_solver_fwd

        self.recorder = None

        # Get GLM Coefficients and explicit/implicit boolean for integration
        (self.GLM_A, self.GLM_B, self.GLM_U, self.GLM_V,
         self.explicit) = get_integration_method(self.method)

        #  WHATISTHIS
        self.dt = 1e-7

        # Initializing properties for variables
        self.parameter_dict = {}
        self.state_dict = {}
        self.f2s_dict = {}
        self.output_state_name_dict = {}
        self.field_output_dict = {}
        self.profile_output_dict = {}
        self.IC_dict = {}

        # Initializing lists of variables
        self.profile_states = []
        self.profile_outputs = []
        self.output_state_list = []
        self.state_output_tuple = []
        self.state_output_name_tuple = []

        # Boolean on whether all defined IC's or parameters are fixed
        self.all_fixed_IC = True
        self.all_fixed_parameters = True

        # Initializing properties of time settings
        self.times = {}

        # For profile output
        self.profile_outputs_system = None
        self.profile_outputs_bool = True

        # True if parameters exist
        self.param_bool = False

        # Creating matplotlib figure if user asks for plot
        if self.visualization == 'during' or self.visualization == 'end':
            self.ongoingplot = plt.figure()
        else:
            self.ongoingplot = None

    def post_setup_init(self):
        """
        Performs tasks needed after "def setup".
        """
        # FORWARD INTEGRATION SETUP
        # Shape of Field-output
        for key in self.field_output_dict:
            state_name = self.field_output_dict[key]['state_name']
            self.field_output_dict[key]['shape'] = self.state_dict[state_name]['shape']
            self.field_output_dict[key]['num'] = np.prod(
                self.state_dict[state_name]['shape'])

        # Find ODE system and Profile output system type
        self.OStype = self.ode_system.system_type
        if self.profile_outputs_bool:
            self.PStype = self.profile_outputs_system.system_type

            for state_name in self.state_dict:
                if state_name not in self.output_state_list:
                    self.output_state_list.append(state_name)

        # Some preprocessing of states
        self.all_states_num = 0
        self.state_output_tuple = tuple(self.state_output_tuple)
        self.state_output_name_tuple = tuple(self.state_output_name_tuple)

        for key in self.state_dict:
            sd = self.state_dict[key]

            # vector index of state for all states concatenated.
            sd['indexes'] = (self.all_states_num, self.all_states_num + sd['num'])

            # Counting total number of state values
            self.all_states_num += sd['num']

            # Shape of state when vectorized accross stages
            if type(sd['shape']) == int:
                if sd['shape'] == 1:
                    sd['nn_shape'] = (len(self.GLM_A),)
                else:
                    sd['nn_shape'] = (len(self.GLM_A), sd['shape'])
            elif type(sd['shape']) == tuple:
                sd['nn_shape'] = (len(self.GLM_A),) + sd['shape']

            if self.explicit:
                # Shape of state when vectorized accross stages (for explicit, it is always one stage)
                if type(sd['shape']) == int:
                    if sd['shape'] == 1:
                        sd['nn_shape_exp'] = (1,)
                    else:
                        sd['nn_shape_exp'] = (1, sd['shape'])
                elif type(sd['shape']) == tuple:
                    sd['nn_shape_exp'] = (1,) + sd['shape']

            if sd['output_bool']:
                # Shape of state when vectorized accross timesteps (when set as output only)
                if type(sd['shape']) == int:
                    if sd['shape'] == 1:
                        sd['output_shape'] = (self.num_steps+1,)
                    else:
                        sd['output_shape'] = (self.num_steps+1, sd['shape'])
                elif type(sd['shape']) == tuple:
                    sd['output_shape'] = (self.num_steps+1,) + sd['shape']

                sd['output_num'] = np.prod(sd['output_shape'])

            # If any of the initial conditions are not fixed, not all initial conditions are fixed
            if sd['fixed_input'] == False:
                self.all_fixed_IC = False

        # Number of stages
        self.num_stages = len(self.GLM_A)  # Number of stages (4 for RK4)
        self.GLM_C = self.GLM_A.dot(np.ones((self.num_stages, 1)))
        if self.explicit:
            self.GLM_C_minus = 1.0 - self.GLM_C

        # Paremeters
        for key in self.parameter_dict:
            if self.parameter_dict[key]['dynamic'] == True:
                # Paremeter vectorized stage shape
                if type(self.parameter_dict[key]['shape']) == int:
                    if self.parameter_dict[key]['shape'] == 1:
                        self.parameter_dict[key]['nn_shape'] = (
                            self.num_stages,)
                    else:
                        self.parameter_dict[key]['nn_shape'] = (
                            self.num_stages, self.parameter_dict[key]['shape'])
                elif type(self.parameter_dict[key]['shape']) == tuple:
                    self.parameter_dict[key]['nn_shape'] = (
                        self.num_stages,) + self.parameter_dict[key]['shape']

                # If explicit, only vectorize accross one stage
                if self.explicit:
                    if type(self.parameter_dict[key]['shape']) == int:
                        if self.parameter_dict[key]['shape'] == 1:
                            self.parameter_dict[key]['nn_shape_exp'] = (
                                1,)
                        else:
                            self.parameter_dict[key]['nn_shape_exp'] = (
                                1, self.parameter_dict[key]['shape'])
                    elif type(self.parameter_dict[key]['shape']) == tuple:
                        self.parameter_dict[key]['nn_shape_exp'] = (
                            1,) + self.parameter_dict[key]['shape']

                tempeye = sp.eye(self.parameter_dict[key]['num'], format='csc')
                self.parameter_dict[key]['stage2state_transform_s'] = sp.kron(
                    1.0 - self.GLM_C, tempeye)

                self.parameter_dict[key]['stage2state_transform_s+'] = sp.kron(
                    self.GLM_C, tempeye)

                tempeye = np.eye(self.parameter_dict[key]['num'])
                self.parameter_dict[key]['stage2state_transform_d'] = np.kron(
                    1.0 - self.GLM_C, tempeye)
                self.parameter_dict[key]['stage2state_transform_d+'] = np.kron(
                    self.GLM_C, tempeye)

            # If any of the parameters are not fixed, not all parameters are fixed
            if self.parameter_dict[key]['fixed_input'] == False:
                self.all_fixed_parameters = False

    def create_solver_model(self, profile_parameters=None, ODE_parameters=None):
        """
        Method that returns integrator system
        """
        # After setup, run post setup initialization
        (numnodes, numnodes_p) = self.post_setup_init()

        self.numnodes = numnodes
        self.numnodes_p = numnodes_p

        # Creating Problems
        self.ode_system.create(self.numnodes, 'O', parameters=ODE_parameters)
        self.ode_system.dt = self.dt
        if self.profile_outputs_bool == True:
            self.profile_outputs_system.create(self.numnodes_p, 'P', parameters=profile_parameters)

        # Returns integrator system
        return self.get_solver_model()

    def get_solver_model(self):
        """
        method gets overwritten by proper solver.
        """
        raise NotImplementedError('get_solver_model not implemented')


    def check_partials(self, sys):
        """
        check partial derivatives of system

        Parameters
        ----------
            sys: list
                Three accepted values: ['ODESystem'], ['ProfileSystem','ODESystem'],['ProfileSystem'].
                Checks partials for either both or one of ODEsystem or ProfileSystem. ONLY WORKS FOR NON-NATIVE Systems.
        """
        # Check partials for profile output system
        if 'ODESystem' in sys:
            of_list = []
            for key in self.state_dict:
                of_list.append(self.state_dict[key]['f_name'])

            wrt_list = []
            for key in self.parameter_dict:
                wrt_list.append(key)

            for key in self.state_dict:
                wrt_list.append(key)

            # Check_totals of systems
            self.ode_system.check_totals(in_of=of_list, in_wrt=wrt_list)

        # Check partials for profile output system
        if 'ProfileSystem' in sys:
            of_list = []
            wrt_list = []
            for key in self.profile_output_dict:
                of_list.append(key)
                wrt_list.append(self.profile_output_dict[key]['state_name'])

            # Check_totals of systems
            self.profile_outputs_system.check_totals(
                in_of=of_list, in_wrt=wrt_list)
