import csdl
import time


class ProfileComp(csdl.CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('parameter_dict')
        self.parameters.declare('IC_dict')
        self.parameters.declare('times')
        self.parameters.declare('state_dict')
        self.parameters.declare('profile_output_dict')
        self.parameters.declare('f2s_dict')
        self.parameters.declare('misc')
        self.parameters.declare('define_dict')

        self.parameters.declare('profile_outputs_system')

    def define(self):
        self.parameter_dict = self.parameters['parameter_dict']
        self.IC_dict = self.parameters['IC_dict']
        self.times = self.parameters['times']
        self.state_dict = self.parameters['state_dict']
        self.profile_output_dict = self.parameters['profile_output_dict']
        self.f2s_dict = self.parameters['f2s_dict']
        self.misc = self.parameters['misc']
        self.po_system = self.parameters['profile_outputs_system']
        self.define_dict = self.parameters['define_dict']

        self.num_steps = self.misc['num_steps']
        self.num_stages = self.misc['num_stages']

        self.to_compute_derivatives = []

        # Inputs: State
        # Outputs: Profile_output_dict

        for key in self.define_dict['inputs']:
            dd = self.define_dict['inputs'][key]
            # print('ode_comp input: ', dd['name'], key)
            self.add_input(**dd)
        for key in self.define_dict['outputs']:
            dd = self.define_dict['outputs'][key]
            self.add_output(**dd)
        for partials in self.define_dict['partials']:
            self.set_partials(**partials)

        self.of_list = []
        self.wrt_list = []
        for key in self.state_dict:
            # Inputs
            self.wrt_list.append(key)
        for key in self.parameter_dict:
            # Inputs
            self.wrt_list.append(key)

        for key in self.profile_output_dict:
            # Outputs
            self.of_list.append(key)

    def compute(self, inputs, outputs):

        # Setting variables to PO system
        run_dict = {}
        for key in self.state_dict:
            state_name = key
            sd = self.state_dict[state_name]
            run_dict[state_name] = inputs[sd['meta_name']]
        for key in self.parameter_dict:
            param_name = key
            pd = self.parameter_dict[param_name]
            run_dict[param_name] = inputs[param_name]

        # Running PO system
        P = self.po_system.run_model(run_dict, self.of_list)

        # Setting outputs
        for key in self.profile_output_dict:
            outputs[key] = P[key]

    def compute_derivatives(self, inputs, partials):

        # start = time.time()
        # Setting variables to PO system
        run_dict = {}
        for key in self.state_dict:
            state_name = key
            sd = self.state_dict[state_name]
            run_dict[state_name] = inputs[sd['meta_name']]
        for key in self.parameter_dict:
            param_name = key
            pd = self.parameter_dict[param_name]
            run_dict[param_name] = inputs[param_name]

        self.po_system.set_vars(run_dict)
        D = self.po_system.compute_total_derivatives(
            self.of_list, self.wrt_list, approach='SB')

        for (key, input_name) in self.to_compute_derivatives:
            if input_name in self.state_dict:
                sd = self.state_dict[input_name]
                partials[key, sd['meta_name']] = D[key][input_name]
            elif input_name in self.parameter_dict:
                pd = self.parameter_dict[input_name]
                partials[key, input_name] = D[key][input_name]
            else:
                raise KeyError(f'dev error: cannot find derivative key for {input_name}')

        # end = time.time()
        # print(end - start)

    def set_partials(self, of, wrt, wrt_real=None):

        if self.po_system.system_type == 'NS':

            if self.po_system.partial_properties[of][wrt_real]['type'] == 'empty':
                return

            self.to_compute_derivatives.append((of, wrt_real))

            if self.po_system.partial_properties[of][wrt_real]['type'] == 'std':
                self.declare_derivatives(of, wrt)

            elif self.po_system.partial_properties[of][wrt_real]['type'] == 'row_col':
                rows_declare = self.po_system.partial_properties[of][wrt_real]['rows']
                cols_declare = self.po_system.partial_properties[of][wrt_real]['cols']
                self.declare_derivatives(of, wrt, rows=rows_declare, cols=cols_declare)

            elif self.po_system.partial_properties[of][wrt_real]['type'] == 'row_col_val':
                rows_declare = self.po_system.partial_properties[of][wrt_real]['rows']
                cols_declare = self.po_system.partial_properties[of][wrt_real]['cols']
                val_declare = self.po_system.partial_properties[of][wrt_real]['val']
                self.declare_derivatives(
                    of, wrt, rows=rows_declare, cols=cols_declare, val=val_declare)

            elif self.po_system.partial_properties[of][wrt_real]['type'] == 'sparse':
                self.declare_derivatives(of, wrt)

            elif self.po_system.partial_properties[of][wrt_real]['type'] == 'cs_uc':
                self.declare_derivatives(of, wrt)

        elif self.po_system.system_type == 'OM':
            self.to_compute_derivatives.append((of, wrt_real))
            self.declare_derivatives(of, wrt)
