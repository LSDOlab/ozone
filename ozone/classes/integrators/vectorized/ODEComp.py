import csdl
import numpy as np
import time


class ODEComp(csdl.CustomExplicitOperation):
    """
    This component computes f(Y) for solver-based approach
    """

    def initialize(self):
        self.parameters.declare('parameter_dict')
        self.parameters.declare('IC_dict')
        self.parameters.declare('times')
        self.parameters.declare('state_dict')
        self.parameters.declare('stage_dict')
        self.parameters.declare('f2s_dict')
        self.parameters.declare('misc')
        self.parameters.declare('ODE_system')
        self.parameters.declare('define_dict')
        self.parameters.declare('stage_f_dict')
        self.parameters.declare('recorder')

    def define(self):
        self.parameter_dict = self.parameters['parameter_dict']
        self.IC_dict = self.parameters['IC_dict']
        self.times = self.parameters['times']
        self.state_dict = self.parameters['state_dict']
        self.stage_dict = self.parameters['stage_dict']
        self.f2s_dict = self.parameters['f2s_dict']
        self.misc = self.parameters['misc']
        self.ode_system = self.parameters['ODE_system']
        self.define_dict = self.parameters['define_dict']
        self.stage_f_dict = self.parameters['stage_f_dict']
        self.recorder = self.parameters['recorder']

        self.num_steps = self.misc['num_steps']
        self.num_stages = self.misc['num_stages']

        self.to_compute_pair = []

        # precomputed arguments of inputs, outputs and partials
        for key in self.define_dict['inputs']:
            dd = self.define_dict['inputs'][key]
            # print('ode_comp input: ', dd['name'], key)
            self.add_input(**dd)
        for key in self.define_dict['outputs']:
            dd = self.define_dict['outputs'][key]
            self.add_output(**dd)
        for partials in self.define_dict['partials']:
            self.set_partials(**partials)

        # # Inputs: Stages
        self.input_list = []
        for key in self.stage_dict:
            state_name = self.stage_dict[key]['state_name']
            self.input_list.append(state_name)

        # # inputs: Parameters
        for key in self.parameter_dict:
            self.input_list.append(key)

        # # Outputs: F
        self.output_list = []
        for key in self.f2s_dict:
            self.output_list.append(key)

    def compute(self, inputs, outputs):
        """
        Computes ODE funcition F = f(Y)
        """
        run_dict = {}
        for key in self.stage_dict:
            state_name = self.stage_dict[key]['state_name']
            run_dict[state_name] = inputs[key].reshape(
                self.state_dict[state_name]['nn_shape'])
        for key in self.parameter_dict:
            proxy_name = self.parameter_dict[key]['proxy_name']
            run_dict[key] = inputs[proxy_name]

        # THIS IS BASICALLY output = ODE function f(Y)
        P = self.ode_system.run_model(run_dict, self.output_list)

        for f_key in P:
            stage_f_name = self.stage_f_dict[f_key]['res_name']
            outputs[stage_f_name] = P[f_key]

    def compute_derivatives(self, inputs, partials):
        """
        Computes partial w.r.t to inputs
        """
        set_dict = {}
        for key in self.stage_dict:
            state_name = self.stage_dict[key]['state_name']
            set_dict[state_name] = inputs[key].reshape(
                self.state_dict[state_name]['nn_shape'])

        for key in self.parameter_dict:
            proxy_name = self.parameter_dict[key]['proxy_name']
            set_dict[key] = inputs[proxy_name]

        self.ode_system.set_vars(set_dict)
        D = self.ode_system.compute_total_derivatives(
            self.output_list, self.input_list, approach='SB')

        for (f_key, wrt_real) in self.to_compute_pair:
            stage_f_name = self.stage_f_dict[f_key]['res_name']
            if wrt_real in self.state_dict:
                stage_name = self.state_dict[wrt_real]['stage_name']
                partials[stage_f_name, stage_name] = D[f_key][wrt_real]

            elif wrt_real in self.parameter_dict:
                proxy_name = self.parameter_dict[wrt_real]['proxy_name']
                partials[stage_f_name, proxy_name] = D[f_key][wrt_real]

    def set_odesys(self, ode_sys):
        self.ode_system = ode_sys

    def set_partials(self, of, wrt, of_real=None, wrt_real=None):

        if self.ode_system.system_type == 'NS':
            if self.ode_system.partial_properties[of_real][wrt_real]['type'] == 'std':
                self.declare_derivatives(of, wrt)
                self.to_compute_pair.append((of_real, wrt_real))

            elif self.ode_system.partial_properties[of_real][wrt_real]['type'] == 'row_col':
                rows_declare = self.ode_system.partial_properties[of_real][wrt_real]['rows']
                cols_declare = self.ode_system.partial_properties[of_real][wrt_real]['cols']
                self.declare_derivatives(of, wrt, rows=rows_declare, cols=cols_declare)
                self.to_compute_pair.append((of_real, wrt_real))

            elif self.ode_system.partial_properties[of_real][wrt_real]['type'] == 'row_col_val':
                rows_declare = self.ode_system.partial_properties[of_real][wrt_real]['rows']
                cols_declare = self.ode_system.partial_properties[of_real][wrt_real]['cols']
                val_declare = self.ode_system.partial_properties[of_real][wrt_real]['vals']
                self.declare_derivatives(of, wrt, rows=rows_declare, cols=cols_declare, val=val_declare)

            elif self.ode_system.partial_properties[of_real][wrt_real]['type'] == 'sparse':
                self.declare_derivatives(of, wrt)
                self.to_compute_pair.append((of_real, wrt_real))

            elif self.ode_system.partial_properties[of_real][wrt_real]['type'] == 'cs_uc':
                self.declare_derivatives(of, wrt)
                self.to_compute_pair.append((of_real, wrt_real))

            elif self.ode_system.partial_properties[of_real][wrt_real]['type'] == 'empty':
                pass

        elif self.ode_system.system_type == 'OM':
            self.declare_derivatives(of, wrt)
            self.to_compute_pair.append((of_real, wrt_real))
