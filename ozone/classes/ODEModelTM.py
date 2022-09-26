from csdl import CustomExplicitOperation, Model, custom
from ozone.classes.integrators.utils import lin_interp


class ODEModelTM(Model):
    """
    Model containting explicit component containing time-marking intergrator
    """

    def define(self):

        input_var_list = []

        # INPUTS: Note that order of input declaration is EXACTLY the same as in the explicit component.
        # this is the only way it will work

        for key in self.integrator.parameter_dict:
            if self.integrator.parameter_dict[key]['dynamic'] == False:
                tempvar = self.declare_variable(
                    key, shape=self.integrator.parameter_dict[key]['shape'])
                input_var_list.append(tempvar)
            else:
                tempvar = self.declare_variable(
                    key, shape=self.integrator.parameter_dict[key]['shape_dynamic'])
                input_var_list.append(tempvar)

        for key in self.integrator.IC_dict:
            tempvar = self.declare_variable(key, shape=self.integrator.IC_dict[key]['shape'])
            input_var_list.append(tempvar)

        if self.integrator.times['type'] == 'step_vector':
            tempvar = self.declare_variable(
                self.integrator.times['name'], shape=self.integrator.num_steps)
            input_var_list.append(tempvar)

        for i, key in enumerate(self.integrator.field_output_dict):

            coef_name = self.integrator.field_output_dict[key]['coefficients_name']
            if i == 0:
                coef_list = [coef_name]
                tempvar = self.declare_variable(coef_name, shape=self.integrator.num_steps+1)
                input_var_list.append(tempvar)

            elif coef_name not in coef_list:
                coef_list.append(coef_name)
                tempvar = self.declare_variable(coef_name, shape=self.integrator.num_steps+1)
                input_var_list.append(tempvar)

        # Add custom operation and feed in the inputs declared above.
        # Recieve tuple of outputs. Also must be in same order as explicit component
        odeec = ODEModelComp()
        odeec.add_ODEProb(self.integrator)
        output_var_tup = custom(*input_var_list, op=odeec)

        # List of output names corresponding to above
        output_order_list = []
        for i, key in enumerate(self.integrator.field_output_dict):
            output_order_list.append(key)
        for key in self.integrator.profile_output_dict:
            output_order_list.append(key)
        for key in self.integrator.state_output_tuple:
            output_order_list.append(self.integrator.state_dict[key]['output_name'])

        # Register the outputs
        if type(output_var_tup) == type((1, 2)):
            for i, out_var in enumerate(output_var_tup):
                self.register_output(output_order_list[i], out_var)
        else:
            self.register_output(output_order_list[0], output_var_tup)

    def add_ODEProb(self, ODEint):
        self.integrator = ODEint


class ODEModelComp(CustomExplicitOperation):
    """
    The explicit model containing the time-marching integrator.
    """

    # Add inputs and outputs
    def define(self):

        # Parameter inputs
        for key in self.integrator.parameter_dict:
            if self.integrator.parameter_dict[key]['dynamic'] == False:
                self.add_input(
                    key, shape=self.integrator.parameter_dict[key]['shape'])
            else:
                self.add_input(
                    key, shape=self.integrator.parameter_dict[key]['shape_dynamic'])

        # Initial condition inputs
        for key in self.integrator.IC_dict:
            self.add_input(key, shape=self.integrator.IC_dict[key]['shape'])

        # time_vector inputs
        if self.integrator.times['type'] == 'step_vector':
            self.add_input(
                self.integrator.times['name'], shape=self.integrator.num_steps)

        # Field outputs and coefficient inputs
        for i, key in enumerate(self.integrator.field_output_dict):
            self.add_output(
                key, shape=self.integrator.field_output_dict[key]['shape'])

            coef_name = self.integrator.field_output_dict[key]['coefficients_name']
            if i == 0:
                coef_list = [coef_name]
                self.add_input(coef_name, shape=self.integrator.num_steps+1)

            elif coef_name not in coef_list:
                coef_list.append(coef_name)
                self.add_input(coef_name, shape=self.integrator.num_steps+1)

        # Profile outputs
        for key in self.integrator.profile_output_dict:
            self.add_output(
                key, shape=self.integrator.profile_output_dict[key]['shape'])

        # State outputs:
        for key in self.integrator.state_output_tuple:
            sd = self.integrator.state_dict[key]
            self.add_output(sd['output_name'], shape=sd['output_shape'])

        self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):
        for key in inputs:
            if key in self.integrator.parameter_dict:
                pd = self.integrator.parameter_dict[key]
                pd['val'] = inputs[key]

                if pd['dynamic'] == True:
                    pd['val_nodal'] = lin_interp(
                        pd['val'], self.integrator.GLM_C, self.integrator.num_steps, pd['nn_shape'])

            elif key in self.integrator.IC_dict:
                self.integrator.IC_dict[key]['val'] = inputs[key]
            elif key == self.integrator.times['name']:
                self.integrator.times['val'] = inputs[key]
            else:
                for field_key in self.integrator.field_output_dict:
                    if self.integrator.field_output_dict[field_key]['coefficients_name'] == key:
                        self.integrator.field_output_dict[field_key]['coefficients'] = inputs[key]

        # Main integration:
        self.integrator.integrate_ODE()

        for key in outputs:
            if key in self.integrator.profile_output_dict:
                outputs[key] = self.integrator.profile_output_dict[key]['val']
            elif key in self.integrator.field_output_dict:
                outputs[key] = self.integrator.field_output_dict[key]['val']
            elif key in self.integrator.state_output_name_tuple:
                state_name = self.integrator.output_state_name_dict[key]['state_name']
                sd = self.integrator.state_dict[state_name]
                outputs[sd['output_name']] = sd['y_out']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # ------------REALLY STRANGE PRE PROCESSING WE HAVE TO DO: -------------
        # If check or compute totals, it calls compute_jacvec_product twice for some reason
        # First time has an empty d_inputs vector which breaks the code. Second time seems to be the correct one.
        # Therefore, when the first computejvp is called (when d_inputs has zero keys), we return.
        no_key = True
        for key in d_inputs:
            no_key = False
        if no_key == True:
            return
        # ---------------------------------------------------------------------------

        if mode == 'rev':
            # Main adjoint calculation:
            d_inputs_return = self.integrator.compute_JVP(d_inputs, d_outputs)
            for key in d_inputs_return:
                d_inputs[key] = d_inputs_return[key]
        elif mode == 'fwd':
            pass

    def add_ODEProb(self, ODEint):
        self.integrator = ODEint
