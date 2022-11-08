import time
from ozone.classes.integrators.IntegratorBase import IntegratorBase
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.linalg as ln
import numpy as np
import matplotlib.pyplot as plt

from ozone.classes.integrators.vectorized.ODEComp import ODEComp
from ozone.classes.integrators.vectorized.MainGroup import VectorBasedGroup


class VectorBased(IntegratorBase):

    def post_setup_init(self):
        super().post_setup_init()

        self.num_stage_time = self.num_steps*self.num_stages

        self.stage_dict = {}
        self.stage_f_dict = {}

        for key in self.state_dict:

            # Creating dict for stages corresponding to each state:
            stage_name = 'stage__' + key  # name of stage
            stage_f_name = 'stage__' + self.state_dict[key]['f_name']  # name of f used in the residual
            state_f_name = 'state__' + self.state_dict[key]['f_name']  # name of f used from the output to the state comp
            meta_name = 'state__' + key
            h_name = key + '_stepvector'  # name of stepsize vector

            # Dictionary for bookkeeping
            self.stage_dict[stage_name] = {
                'num': self.state_dict[key]['num']*self.num_steps*self.num_stages,
                'state_name': key,
                'h_name': h_name,
                'stage_comp_out_name': stage_name + '_2'}
            self.stage_f_dict[self.state_dict[key]['f_name']] = {}
            self.stage_f_dict[self.state_dict[key]['f_name']]['res_name'] = stage_f_name
            self.stage_f_dict[self.state_dict[key]['f_name']]['state_name'] = state_f_name
            self.stage_f_dict[self.state_dict[key]['f_name']]['state_key'] = key

            self.state_dict[key]['h_name'] = h_name
            self.state_dict[key]['stage_name'] = stage_name
            self.state_dict[key]['meta_name'] = meta_name

            # Creating GLM matrices corresponding to each state:
            A_kron = sp.kron(sp.csc_matrix(self.GLM_A), sp.eye(
                self.state_dict[key]['num'], format='csc'), format='csr')
            B_kron = sp.kron(sp.csc_matrix(self.GLM_B), sp.eye(
                self.state_dict[key]['num'], format='csc'), format='csr')
            U_kron = sp.kron(sp.csc_matrix(self.GLM_U), sp.eye(
                self.state_dict[key]['num'], format='csc'), format='csr')
            V_kron = sp.kron(sp.csc_matrix(self.GLM_V), sp.eye(
                self.state_dict[key]['num'], format='csc'), format='csr')
            ImV_full = sp.eye((self.num_steps+1)*self.state_dict[key]['num'], format='csc') - sp.kron(
                sp.eye(self.num_steps+1, k=-1, format='csc'), V_kron, format='csc')
            ImV_full_inv = spln.inv(ImV_full)

            self.state_dict[key]['A_full'] = sp.kron(
                sp.eye(self.num_steps, format='csc'), A_kron, format='csc')
            self.state_dict[key]['U_full'] = sp.kron(
                sp.eye(self.num_steps, n=self.num_steps+1, format='csc'), U_kron, format='csc')
            self.state_dict[key]['B_full'] = sp.kron(sp.eye(
                self.num_steps + 1, n=self.num_steps, k=-1, format='csc'), B_kron, format='csc')
            self.state_dict[key]['ImV_inv'] = ImV_full_inv
            self.state_dict[key]['UImV_inv'] = self.state_dict[key]['U_full']*ImV_full_inv

            # Defining Shapes
            if type(self.state_dict[key]['shape']) == int:
                if self.state_dict[key]['shape'] == 1:
                    self.state_dict[key]['nn_shape'] = (self.num_stage_time,)
                    self.state_dict[key]['output_shape'] = (self.num_steps+1,)
                else:
                    self.state_dict[key]['nn_shape'] = (
                        self.num_stage_time, self.state_dict[key]['shape'])
                    self.state_dict[key]['output_shape'] = (
                        self.num_steps+1, self.state_dict[key]['shape'])
            elif type(self.state_dict[key]['shape']) == tuple:
                self.state_dict[key]['nn_shape'] = (
                    self.num_stage_time,) + self.state_dict[key]['shape']
                self.state_dict[key]['output_shape'] = (
                    self.num_steps+1,) + self.state_dict[key]['shape']

            self.state_dict[key]['nn_guess'] = np.linspace(
                self.state_dict[key]['guess'][0], 
                self.state_dict[key]['guess'][1], 
                num = np.prod(self.state_dict[key]['nn_shape'])).reshape(self.state_dict[key]['nn_shape'])

        # Parameter Meta Names and shapes
        for key in self.parameter_dict:
            self.parameter_dict[key]['proxy_name'] = 'proxy_'+key
            if self.parameter_dict[key]['dynamic'] == False:
                pass
            else:
                if type(self.parameter_dict[key]['shape_dynamic']) == type((1, 2)):
                    proxy_shape = list(self.parameter_dict[key]['shape_dynamic'])
                else:
                    proxy_shape = [self.parameter_dict[key]['shape_dynamic']]
                proxy_shape[0] = self.num_stages*(proxy_shape[0]-1)
                proxy_shape = tuple(proxy_shape)
                self.parameter_dict[key]['proxy_shape'] = proxy_shape

        # IC Meta Names and shapes
        for key in self.IC_dict:
            state_name = self.IC_dict[key]['state_name']
            self.IC_dict[key]['meta_name'] = key+'_vector'
            self.IC_dict[key]['vector_shape'] = (
                self.state_dict[state_name]['num']*(self.num_steps+1),)

        # For the csdl explicit components, we need to track the ORDER in which we declare inputs and outputs for each component
        # This is so annoyingly complex but currently cannot think of a better way
        self.var_order_name = {}
        comp_list = ['InputProcessComp', 'ODEComp', 'StageComp', 'StateComp', 'FieldComp', 'ProfileComp']
        for comp_name in comp_list:
            # We have a dictionary for each component that we feed through
            var_dict = self.order_variables(comp_name)
            self.var_order_name[comp_name] = var_dict

        # ===================== UNCOMMENT TO SEE VARIABLE ORDERING HIERARCHY =====================
        # print('VARIABLE DICTIONARY HIERARCHY')
        # for comp_name in comp_list:
        #     print()
        #     print(f'------------------------{comp_name}------------------------')
        #     for io in self.var_order_name[comp_name]:
        #         vonk = self.var_order_name[comp_name][io]
        #         if io != 'partials':
        #             print(f'---{io}------------------------')
        #             for key in vonk:
        #                 a_in = vonk[key]
        #                 name = a_in['name']
        #                 print(f'       DICTKEY: {key},    VARNAME: {name}')
        # ===================== UNCOMMENT TO SEE VARIABLE ORDERING HIERARCHY =====================

        self.num_stage_time = self.num_steps*self.num_stages

        return self.num_stage_time, self.num_steps+1

    def order_variables(self, comp_name):
        """
        Given the name of the component, constructs dictionary of
        the variable name string and shape
        """

        input_dict_return = {}
        output_dict_return = {}
        partial_return = []

        # Input Processing Comp:
        for key in self.parameter_dict:
            proxy_name = self.parameter_dict[key]['proxy_name']
            if self.parameter_dict[key]['dynamic'] == False:
                # Input
                if comp_name == 'InputProcessComp':
                    input_dict_return[key] = {'name': key, 'shape': self.parameter_dict[key]['shape']}

                # Output
                if comp_name == 'InputProcessComp':
                    output_dict_return[key] = {'name': proxy_name, 'shape': self.parameter_dict[key]['shape']}
                if comp_name == 'ODEComp':
                    input_dict_return[key] = {'name': proxy_name, 'shape': self.parameter_dict[key]['shape']}

                # Partials: dparam/dparam = identity
                if comp_name == 'InputProcessComp':
                    row_col = np.arange(0, self.parameter_dict[key]['num'])
                    val = np.ones(self.parameter_dict[key]['num'])
                    partial_return.append({'of': proxy_name, 'wrt': key, 'rows': row_col, 'cols': row_col, 'val': val})

            else:
                # Input
                if comp_name == 'InputProcessComp':
                    input_dict_return[key] = {'name': key, 'shape': self.parameter_dict[key]['shape_dynamic']}

                # Output
                proxy_shape = self.parameter_dict[key]['proxy_shape']
                if comp_name == 'InputProcessComp':
                    output_dict_return[key] = {'name': proxy_name, 'shape': proxy_shape}
                if comp_name == 'ODEComp':
                    input_dict_return[key] = {'name': proxy_name, 'shape': proxy_shape}

                # Partials:dproxy_param/dparam = block diag identity
                if comp_name == 'InputProcessComp':
                    # Interpolated partials:
                    rows = []
                    cols = []
                    vals = []
                    for i in range(self.num_steps):
                        for j in range(self.num_stages):
                            # i is the timestep #
                            # j is the stage #

                            # if i == self.num_steps-1:
                            #     # if last timestep, there is no linear interpolation
                            #     row_temp = np.arange(0, self.parameter_dict[key]['num'])
                            #     col_temp = np.arange(0, self.parameter_dict[key]['num'])
                            #     vals_temp = np.ones(self.parameter_dict[key]['num'])

                            #     row_temp += i * self.parameter_dict[key]['num']*self.num_stages
                            #     col_temp += i*self.parameter_dict[key]['num']

                            #     row_temp += j*self.parameter_dict[key]['num']

                            #     rows.extend(row_temp)
                            #     cols.extend(col_temp)
                            #     vals.extend(vals_temp)

                            #     continue

                            # PARTIALS FOR PARAMETER i
                            # diagonals of current stage and timestep
                            row_temp = np.arange(0, self.parameter_dict[key]['num'])
                            col_temp = np.arange(0, self.parameter_dict[key]['num'])

                            # adjust row and column for timestep
                            row_temp += i * self.parameter_dict[key]['num']*self.num_stages
                            col_temp += i * self.parameter_dict[key]['num']

                            # adjust row for stage
                            row_temp += j*self.parameter_dict[key]['num']
                            vals_temp = np.ones(self.parameter_dict[key]['num'])*(1.0-self.GLM_C[j])

                            rows.extend(row_temp)
                            cols.extend(col_temp)
                            vals.extend(vals_temp)

                            # PARTIALS FOR PARAMETER i+1
                            # adjust column the i+1'th parameter
                            col_temp += self.parameter_dict[key]['num']

                            # adjust row for stage
                            vals_temp = np.ones(self.parameter_dict[key]['num'])*(self.GLM_C[j])

                            rows.extend(row_temp)
                            cols.extend(col_temp)
                            vals.extend(vals_temp)

                    # print(sp.csc_matrix((vals, (rows, cols))).toarray())
                    # print(proxy_shape)
                    partial_return.append({'of': proxy_name, 'wrt': key, 'rows': rows, 'cols': cols, 'val': vals})

        # IC's: inputs and vector outputs
        for key in self.IC_dict:
            # inputs:
            if comp_name == 'InputProcessComp':
                input_dict_return[key] = {'name': key, 'shape': self.IC_dict[key]['shape']}

            # outputs:
            if comp_name == 'InputProcessComp':
                output_dict_return[key] = {'name': self.IC_dict[key]['meta_name'], 'shape': self.IC_dict[key]['vector_shape']}

            # partials:
            if comp_name == 'InputProcessComp':
                rows_cols = list(np.arange(0, self.IC_dict[key]['num']))
                vals = list(np.ones(self.IC_dict[key]['num']))
                partial_return.append({'of': self.IC_dict[key]['meta_name'], 'wrt': key, 'rows': rows_cols, 'cols': rows_cols, 'val': vals})

        # Step_vector: inputs and matrix outputs for each stage
        if self.times['type'] == 'step_vector':
            # Inputs
            if comp_name == 'InputProcessComp':
                input_dict_return[self.times['name']] = {'name': self.times['name'], 'shape': self.num_steps}

        for key in self.stage_dict:
            # Outputs
            state_name = self.stage_dict[key]['state_name']
            flat_shape = (self.state_dict[state_name]['num'] * self.num_stages*self.num_steps,)
            h_name = self.stage_dict[key]['h_name']
            stage_2_name = self.stage_dict[key]['stage_comp_out_name']
            if comp_name == 'InputProcessComp':
                output_dict_return[h_name] = {'name': h_name, 'shape': flat_shape}
            if comp_name == 'ODEComp':
                input_dict_return[key] = {'name': key, 'shape': flat_shape}
            if comp_name == 'StageComp':
                output_dict_return[key] = {'name': stage_2_name, 'shape': flat_shape}

            # Partials:
            if comp_name == 'InputProcessComp':
                rows = []
                cols = []
                vals = list(np.ones(flat_shape))
                for i in range(self.num_stages * self.state_dict[state_name]['num']):
                    cols.extend(np.arange(0, self.num_steps))
                    rows.extend(np.arange(0, flat_shape[0], self.num_stages * self.state_dict[state_name]['num'])+i)
                partial_return.append({'of': h_name, 'wrt': self.times['name'], 'rows': rows, 'cols': cols, 'val': vals})

                # Creating initial IC vector as well
                self.stage_dict[key]['IC_vector_initialize'] = np.zeros(
                    (self.num_steps+1)*self.state_dict[state_name]['num'])

            if comp_name == 'StageComp':
                # IC inputs:
                sd = self.state_dict[state_name]
                icname = sd['IC_name']
                ICname_meta = self.IC_dict[icname]['meta_name']
                input_dict_return[icname] = {'name': ICname_meta, 'shape': self.IC_dict[icname]['vector_shape']}
                # Sparse Jacobian is broken????....
                # partial_return.append({'of': stage_2_name, 'wrt': ICname_meta, 'val': sd['UImV_inv']})
                # Looks like I need to set it to dense...
                partial_return.append({'of': stage_2_name, 'wrt': ICname_meta, 'val': sd['UImV_inv'].toarray()})

                # Times vector inputs:
                # flat_shape = sd['num']*self.num_stages*self.num_steps
                h_name = self.stage_dict[key]['h_name']
                input_dict_return[h_name] = {'name': h_name, 'shape': flat_shape}

                # Finding required rows/cols of partials
                dYdH_coefficient = sd['A_full'] + sd['UImV_inv'] * sd['B_full']
                self.state_dict[state_name]['dYdH_coefficient'] = dYdH_coefficient.toarray()
                partial_return.append({'of': stage_2_name, 'wrt': h_name})

                # f:
                f_name = sd['f_name']
                stage_f_name = self.stage_f_dict[f_name]['res_name']
                input_dict_return[f_name] = {'name': stage_f_name, 'shape': (self.num_steps) * self.num_stages*sd['num']}
                partial_return.append({'of': stage_2_name, 'wrt': stage_f_name})

        for key in self.f2s_dict:
            state_name = self.f2s_dict[key]
            stage_f_name = self.stage_f_dict[key]['res_name']
            # print(comp_name, key, state_name)

            if comp_name == 'ODEComp':
                # self.add_output(key, shape=(self.num_steps) * self.num_stages*self.state_dict[state_name]['num'])
                output_dict_return[key] = {'name': stage_f_name, 'shape': ((self.num_steps) * self.num_stages*self.state_dict[state_name]['num'],)}

                for param in self.parameter_dict:
                    proxy_name = self.parameter_dict[param]['proxy_name']
                    # self.set_partials(key, proxy_name, wrt_real=param)
                    partial_return.append({'of': stage_f_name, 'wrt': proxy_name, 'of_real': key, 'wrt_real': param})

                for stage in self.stage_dict:
                    # self.set_partials(key, stage, wrt_real=state_name)
                    # print(comp_name, key, stage, state_name)
                    partial_return.append({'of': stage_f_name, 'wrt': stage, 'of_real': key,  'wrt_real': self.stage_dict[stage]['state_name']})

        for key in self.output_state_list:
            # state output:
            state_name = key
            sd = self.state_dict[state_name]
            stage_name = sd['stage_name']
            if comp_name == 'StateComp':
                output_dict_return[key] = {'name': sd['meta_name'], 'shape': sd['output_shape']}

            if comp_name == 'StateComp':

                # IC inputs:
                icname = sd['IC_name']
                icname_meta = self.IC_dict[icname]['meta_name']
                input_dict_return[icname] = {'name': icname_meta, 'shape': self.IC_dict[icname]['vector_shape']}

                # Sparse Jacobian is broken????....
                # partial_return.append({'of': sd['meta_name'], 'wrt': icname_meta, 'val': sd['ImV_inv']})
                # Looks like I need to set it to dense...
                partial_return.append({'of': sd['meta_name'], 'wrt': icname_meta, 'val': sd['ImV_inv'].toarray()})

                # Times vector inputs:
                flat_shape = sd['num']*self.num_stages*self.num_steps
                h_name = self.stage_dict[stage_name]['h_name']
                input_dict_return[h_name] = {'name': h_name, 'shape': flat_shape}
                ImV_invB = sd['ImV_inv']*sd['B_full']
                self.state_dict[state_name]['ImV_invB'] = ImV_invB.toarray()
                partial_return.append({'of': sd['meta_name'], 'wrt': h_name})

                # f:
                f_name = sd['f_name']
                state_f_name = self.stage_f_dict[f_name]['state_name']
                input_dict_return[f_name] = {'name': state_f_name, 'shape': (self.num_steps) * self.num_stages*sd['num']}
                partial_return.append({'of': sd['meta_name'],  'wrt': state_f_name})

        if comp_name == 'FieldComp':
            of_list = []
            wrt_list = []
            for i, key in enumerate(self.field_output_dict):
                state_name = self.field_output_dict[key]['state_name']
                sd = self.state_dict[state_name]

                # Inputs
                wrt_list.append(state_name)
                input_dict_return[state_name] = {'name': sd['meta_name'], 'shape': sd['output_shape']}

                coeff_name = self.field_output_dict[key]['coefficients_name']
                if i == 0:
                    coeff_list = [coeff_name]
                    input_dict_return[self.field_output_dict[key]['coefficients_name']] = {'name': self.field_output_dict[key]['coefficients_name'], 'shape': (self.num_steps+1)}
                elif coeff_name not in coeff_list:
                    coeff_list.append(coeff_name)
                    input_dict_return[self.field_output_dict[key]['coefficients_name']] = {'name': self.field_output_dict[key]['coefficients_name'], 'shape': (self.num_steps+1)}

                # Outputs
                of_list.append(key)
                output_dict_return[key] = {'name': key, 'shape': self.field_output_dict[key]['shape']}

                # Partials
                rows = []
                cols = []
                for t in range(self.num_steps+1):
                    row_temp = np.arange(0, sd['num'])
                    col_temp = np.arange(0, sd['num'])

                    col_temp += t*sd['num']

                    rows.extend(row_temp)
                    cols.extend(col_temp)
                partial_return.append({'of': key, 'wrt': sd['meta_name'], 'rows': rows, 'cols': cols})

        if comp_name == 'ProfileComp':
            if self.profile_outputs_bool:
                for key in self.state_dict:
                    state_name = key
                    sd = self.state_dict[state_name]

                    # Inputs
                    input_dict_return[state_name] = {
                        'name': sd['meta_name'],
                        'shape': sd['output_shape']
                    }

                for key in self.parameter_dict:
                    param_name = key
                    pd = self.parameter_dict[key]

                    # for key2 in pd:
                    #     print(key, key2, pd[key2])

                    if pd['dynamic']:
                        shape = pd['shape_dynamic']
                    else:
                        shape = pd['shape']

                    input_dict_return[param_name] = {
                        'name': key,
                        'shape': shape,
                    }

                for key in self.profile_output_dict:

                    # Outputs
                    output_dict_return[key] = {'name': key, 'shape': self.profile_output_dict[key]['shape']}

                    # Partials
                    for state_name in self.state_dict:
                        sd = self.state_dict[state_name]
                        partial_return.append({'of': key, 'wrt': sd['meta_name'], 'wrt_real': state_name})
                    for param_name in self.parameter_dict:
                        pd = self.parameter_dict[param_name]
                        partial_return.append({'of': key, 'wrt': param_name, 'wrt_real': param_name})

        return {'inputs': input_dict_return, 'outputs': output_dict_return, 'partials': partial_return}

    def get_solver_model(self):

        # return SolverBased group
        component = VectorBasedGroup()
        component.add_ODEProb(self)

        return component
