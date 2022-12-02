import numpy as np
import matplotlib.pyplot as plt
from ozone.classes.integrators.GetIntegrator import get_integrator
from ozone.classes.Wrap import Wrap
from ozone.classes.NativeSystem import NativeSystem
import csdl


class ODEProblem(object):

    def __init__(self,
                 method,
                 approach,
                 num_times,
                 display='default',
                 error_tolerance=0.0000001,
                 visualization=None,
                 dictionary_inputs=None,
                 num_checkpoints=None,
                 implicit_solver_jvp='direct',
                 implicit_solver_fwd='direct'):
        """
        Parameters
        ----------
            method: str
                Numerical method to use for time integration
            approach: str
                Approach to use for numerical integration. Acceptable values are 'time-marching'
            num_steps: int
                Number of timesteps for numerical integration
            display: str
                Settings for what text to display during/after the integration process
            error_tolerance: float
                Error tolerance for implicit stage calculation nonlinear solver
            visualization: float
                Settings for what plot to display during/after the integration process
            dictionary_inputs: dict
                Dictionary of values to pass into the class for use in setup method. Access by: self.dictionary_inputs[key]
            num_checkpoints: int
                Number of checkpoints if time-marching with checkpoints are used. Not used if approach is time-marching, solver-based
                or collocation
        """
        # Approach determines which integrator class to use
        self.approach = approach

        # Integrator class is different depending on approach defined by user.
        IntegratorClass = get_integrator(self.approach)
        # Store integrator
        self.integrator = IntegratorClass(method,
                                          num_times,
                                          display=display,
                                          error_tolerance=error_tolerance,
                                          visualization=visualization,
                                          num_checkpoints=num_checkpoints,
                                          implicit_solver_fwd=implicit_solver_fwd,
                                          implicit_solver_jvp=implicit_solver_jvp)
        self.integrator.approach = approach

        # Set to true when user defines a profile output system in "setup"
        self.profile_outputs_system = None

        # Dictionary inputs that user can use in def setup
        self.num_times = num_times
        self.num_steps = num_times-1  # number of time steps.
        self.dictionary_inputs = dictionary_inputs

        # Recorder for LSDO Dash
        self.recorder = None

    def setup(self):
        """
        Must be defined by the user. Tells the class how the ODE is defined. 
        The following methods must/may be called within this method:
        - self.add_state()              : REQUIRED, called once per state variable              , define a state variable
        - self.add_times()              : REQUIRED, called once                                 , define the time domain
        - self.set_ode_system()         : REQUIRED, called once                                 , define the ODE function

        - self.add_parameter()          : OPTIONAL, called once per parameter variable          , define a parameter variable
        - self.add_field_output()       : OPTIONAL, called once per field output                , define a field output
        - self.add_profile_output()     : OPTIONAL, called once per profile output              , define a profile output
        - self.set_profile_system()     : OPTIONAL, called once if atleast one profile output   , define the profile function

        See examples.
        """

    # Methods to organize variables. Similar to Ozone's ODEfunction's "add" mthods
    def add_parameter(self, parameter_name, shape=1, dynamic=False, fixed_input=False):
        """
        define static/dynamic parameter

        Parameters
        ----------
            parameter_name: str
                Parameter name of upstream component variable to pass into the integrator
            shape: int or Tuple[int]
                Shape of parameter input. If dynamic parameter, shape = (self.num_times, parameter shape at every timestep)
            dynamic: bool
                If True, the parameter input varies at every timestep. Shape must be (self.num_times, parameter shape at every timestep)
            fixed_input: bool
                If True, the parameter is fixed_input. ie. derivatives wrt the parameters are manually set to zero and not computed. This saves computation time.
        """
        # error:
        if not isinstance(shape, (type((1, 2)), type(10))):
            raise ValueError(f'parameter shape must integer or tuple')

        # Tell integrator class that parameters will be used for integration
        self.integrator.param_bool = True

        # Dictionary of parameter properties
        temp_dict = {}
        temp_dict['dynamic'] = dynamic
        if dynamic == True:

            if isinstance(shape, type((1, 2))):
                if shape[0] != self.num_times:
                    raise ValueError('parameter "' + parameter_name +
                                     '": Dynamic parameters must have shape (num_timepoints, ..shape of parameter...). Given shape is ' + str(shape))
            elif isinstance(shape, type(10)):
                if shape != self.num_times:
                    raise ValueError('parameter "' + parameter_name +
                                     '": Dynamic parameters must have shape (num_timepoints, ..shape of parameter...). Given shape is ' + str(shape))
            else:
                raise ValueError(f'shape of parameter {parameter_name} must be an integer or tuple')

            temp_dict['shape_dynamic'] = shape
            temp_dict['num_dynamic'] = np.prod(shape)
            try:
                if isinstance(shape, type((1, 2))):

                    if len(shape) > 1:
                        temp_dict['shape'] = (shape[1:])
                    else:
                        temp_dict['shape'] = 1
                else:
                    temp_dict['shape'] = 1
            except:
                raise ValueError('parameter "' + parameter_name +
                                 '": Dynamic parameters must have shape (num_timepoints, ..shape of parameter...). Given shape is ' + str(shape))
            temp_dict['num'] = np.prod(temp_dict['shape'])
            temp_dict['fixed_input'] = fixed_input
        else:
            temp_dict['shape'] = shape
            temp_dict['num'] = np.prod(shape)
            temp_dict['fixed_input'] = fixed_input

        # Store parameter properties in integrator class
        self.integrator.parameter_dict[parameter_name] = temp_dict

    def add_times(self, shape=1, step_vector=None, fixed_input=False):
        """
        Define time variable

        Parameters
        ----------
            step_vector: str
                A string of the upstream component that defines 1-dimensional array containing the time step at every iteration
            step_vector: iterable
                The name of the csdl variable representing the timestep vector. Must be of size number of timesteps.
                For example, if we want to integrate from 0 ~ 1 seconds with 5 timesteps with a timestep size of 0.2sec, step_vector could be initialized as a csdl variable with
                [0.2, 0.2, 0.2, 0.2, 0.2] as the timestep vector.
            fixed_input: bool
                if True, the time vector is fixed_input. ie. derivatives wrt time value are manually set to zero and not computed. This saves computation time.
        """

        if type(step_vector) is not None:
            self.integrator.times['type'] = 'step_vector'
            self.integrator.times['name'] = step_vector
            self.integrator.times['val'] = None
            self.integrator.times['fixed_input'] = fixed_input
        else:
            raise ValueError('step_vector has not been set.')

    def add_state(self,
                  state_name,
                  f_name,
                  shape=1,
                  initial_condition_name=None,
                  fixed_input=False,
                  output=None,
                  interp_guess=[1.0, 1.0]):
        """
        define state

        Parameters
        ----------
            state_name: str
                State name of upstream component variable to solve for.
            f_name: str
                Name of corresponding function output name used in the ODEsystem
            initial_condition_name: str
                Name of corresponding initial condition variable name from upstream copmonents
            shape: int or Tuple[int]
                Shape of state.
            dynamic: bool
                If True, the parameter input varies at every timestep. Shape must be (self.num_steps, parameter shape at every timestep)
            fixed_input: bool
                If true, the initial condition is fixed_input. ie. derivatives wrt initial conditions are manually set to zero and not computed. This saves computation time.
            output: str
                If solved state is desired as an output to the integrator, specify name of output. Declare this variable name to access in csdl.
            interp_guess: np.array, float
                If solution approach is 'collocation', set an initial guess. Default is 1.0.
        """
        # error:
        if initial_condition_name is None:
            raise ValueError(f'initial condition for {state_name} has not been set.')
        if not isinstance(shape, (type((1, 2)), type(10))):
            raise ValueError(f'state shape must integer or tuple')
        if not isinstance(interp_guess, list):
            # TODO: add more options.
            raise ValueError('interp_guess must be a list with two elements')

        # Dictionary of state properties
        temp_dict = {
            'shape': shape,
            'num': np.prod(shape),
            'IC_name': initial_condition_name,
            'f_name': f_name,
            'f_val': None,
            'val': None,
            'fixed_input': fixed_input,
            'guess': interp_guess}
        self.integrator.state_dict[state_name] = temp_dict
        self.integrator.f2s_dict[f_name] = state_name
        temp_dict = {
            'state_name': state_name,
            'shape': shape,
            'num': np.prod(shape),
            'val': None,
            'fixed_input': fixed_input}
        self.integrator.IC_dict[initial_condition_name] = temp_dict

        if output is not None:
            self.integrator.state_dict[state_name]['output_bool'] = True
            self.integrator.state_dict[state_name]['output_name'] = output
            self.integrator.state_output_tuple.append(state_name)
            self.integrator.state_output_name_tuple.append(output)
            self.integrator.output_state_name_dict[output] = {}
            self.integrator.output_state_name_dict[output]['state_name'] = state_name
            if state_name not in self.integrator.output_state_list:
                self.integrator.output_state_list.append(state_name)
        else:
            self.integrator.state_dict[state_name]['output_bool'] = False

    def add_field_output(self, field_output_name, coefficients_name=None, state_name=None):
        """
        define a field output

        Parameters
        ----------
            field_output_name: str
                Name of field output to be used in downstream systems.
            coefficients_name: str
                Name of upstream component variable containing coefficients for the field output of shape self.numtimes+1
            state_name: str
                Name of corresponding state to take field output of
        """
        # Dictionary of field output properties
        temp_dict = {
            'coefficients_name': coefficients_name,
            'state_name': state_name,
            'val': None,
            'coefficients': None}
        self.integrator.field_output_dict[field_output_name] = temp_dict
        if state_name not in self.integrator.output_state_list:
            self.integrator.output_state_list.append(state_name)

    def add_profile_output(self, profile_output_name, state_name=None, shape=1):
        """
        define a profile output. If called, must be in the setup method. Further, a profile model must be declared using the
        set_profile_system method.

        Parameters
        ----------
            profile_output_name: str
                Name of field output to be used in downstream systems.
            state_name: str
                Name of corresponding state to take profile output of.
            shape: int or Tuple[int]
                Shape of profile output at every timestep.
        """
        # Dictionary of profile output properties
        self.integrator.profile_output_dict[profile_output_name] = {}
        self.integrator.profile_output_dict[profile_output_name]['shape_single'] = shape
        self.integrator.profile_output_dict[profile_output_name]['num_single'] = np.prod(
            shape)
        if shape == 1:
            self.integrator.profile_output_dict[profile_output_name]['shape'] = (
                self.integrator.num_steps+1, shape)
        else:
            self.integrator.profile_output_dict[profile_output_name]['shape'] = (
                self.integrator.num_steps+1,) + shape
        self.integrator.profile_output_dict[profile_output_name]['num'] = np.prod(
            self.integrator.profile_output_dict[profile_output_name]['shape'])
        self.integrator.profile_output_dict[profile_output_name]['state_name'] = state_name
        self.integrator.profile_states.append(state_name)
        self.integrator.profile_outputs.append(profile_output_name)
        # if state_name not in self.integrator.output_state_list:
        #     self.integrator.output_state_list.append(state_name)

    def create_solver_model(self, ODE_parameters=None, profile_parameters=None):
        """
        Create CSDL model containing integrator. "def setup" is called here.

        Parameters:
        ----------
        ODE_parameters: dictionary
            Dictionary containing CSDL parameters to be passed into the ode_system.

        profile_parameters: dictionary
            Dictionary containing CSDL parameters to be passed into the profile_outputs_system if applicable.
        """

        self.setup()

        if self.profile_outputs_system == None:
            self.integrator.profile_outputs_bool = False
        else:
            self.integrator.profile_outputs_system = self.profile_outputs_system
        self.integrator.ode_system = self.ode_system
        return self.integrator.create_solver_model(ODE_parameters=ODE_parameters, profile_parameters=profile_parameters)

    def check_partials(self, system_type):
        """
        check partial derivatives of system

        Parameters
        ----------
            system_type: list
                Three accepted values: ['ODESystem'], ['ProfileSystem','ODESystem'],['ProfileSystem'].
                Checks partials for either both or one of ODEsystem or ProfileSystem. ONLY WORKS FOR NON-NATIVE Systems.
        """
        self.integrator.check_partials(system_type)

    def add_recorder(self, recorder):
        """
        Dashboard recorder from LSDO dashboard
        """
        self.integrator.ode_system.recorder = recorder

    def set_ode_system(self, ode_system, backend='python_csdl_backend'):
        """
        Declare the ODE system. This method MUST be called in the setup method.

        Parameters
        ----------
            ode_system: 
                -NativeSystem or CSDL Model computing dydt = f(x,y)
        """

        # if type(ode_system) == csdl.core.model._CompilerFrontEndMiddleEnd:
        if issubclass(ode_system, csdl.Model):
            self.ode_system = Wrap(ode_system, backend)
        elif issubclass(ode_system, NativeSystem):
            self.ode_system = ode_system()
        else:
            raise TypeError('must be an uninstantiated CSDL Model or uninstantiated NativeSystem')

    def set_profile_system(self, profile_system, backend='python_csdl_backend'):
        """
        Declare the Profile Output system. This method MUST be called in the setup method IF profile outputs are declared.

        Parameters
        ----------
            profile_system: 
                -NativeSystem or CSDL Model computing dydt = f(x,y)
        """

        if issubclass(profile_system, csdl.Model):
            self.profile_outputs_system = Wrap(profile_system, backend)
        elif issubclass(profile_system, NativeSystem):
            self.profile_outputs_system = profile_system()
        else:
            raise TypeError('must be an uninstantiated CSDL Model or uninstantiated NativeSystem')
