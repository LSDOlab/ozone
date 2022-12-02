import numpy as np
import scipy.sparse as sp


class NativeSystem(object):

    def __init__(self):
        self.num_nodes = 0
        self.system_type = 'NS'
        self.dt = 1e-9

        self.input_vals = {}
        self.input_dict = {}
        self.output_vals = {}
        self.output_dict = {}
        self.partial_vals = {}
        self.partial_properties = {}

        self.complex_wrt_dict = {}

        self.num_f_calls = 0
        self.num_vectorized_f_calls = 0
        self.num_df_calls = 0
        self.num_vectorized_df_calls = 0

        self.recorder = None

    def create(self, num_in, type, parameters=None):
        self.num_nodes = num_in

        # Add parameters as attributes
        if parameters is not None:
            self.parameters = {}
            for parameter_key in parameters:
                self.parameters[parameter_key] = parameters[parameter_key]

        # Run setup
        self.setup()

        # Setting the partial derivative properties as specified by user in setup
        for outs in self.output_vals:
            for ins in self.input_vals:
                self.partial_vals[outs] = {}
                self.partial_vals[outs][ins] = None

                if outs in self.partial_properties:
                    if ins in self.partial_properties[outs]:
                        if 'type' in self.partial_properties[outs][ins]:
                            continue
                        else:
                            self.partial_properties[outs][ins]['type'] = 'std'
                    else:
                        self.partial_properties[outs][ins] = {}
                        self.partial_properties[outs][ins]['type'] = 'std'
                else:
                    self.partial_properties[outs] = {}
                    self.partial_properties[outs][ins] = {}
                    self.partial_properties[outs][ins]['type'] = 'std'

    def run_model(self, input_dict, output_vals):
        self.num_vectorized_f_calls += 1
        self.num_f_calls += self.num_nodes
        if self.recorder:
            save_dict = self.get_recorder_data(self.recorder.dash_instance.vars['ozone']['var_names'])
            self.recorder.record(save_dict, 'ozone')

        # Runs model. Can also set variables if needed
        for key in input_dict:
            self.input_vals[key] = input_dict[key]

        outputs = self.output_vals
        inputs = self.input_vals
        self.compute(inputs, outputs)

        return_outputs = {}
        for key in output_vals:
            return_outputs[key] = outputs[key]
        return return_outputs

    def compute_total_derivatives(self, in_of, in_wrt, approach='TM'):

        self.num_vectorized_df_calls += 1
        self.num_df_calls += self.num_nodes
        if self.recorder:
            save_dict = self.get_recorder_data(self.recorder.dash_instance.vars['ozone']['var_names'])
            self.recorder.record(save_dict, 'ozone')

        partials = self.partial_vals
        partial_properties = self.partial_properties
        inputs = self.input_vals
        cs_uc_run_list = []
        complex_wrt_dict = {}

        # for of in in_of:
        #     for wrt in in_wrt:
        #         partials[of, wrt] = np.zeros(self.output_dict(of), self.input_dict(wrt))

        self.compute_partials(inputs, partials)

        if approach == 'TM':
            return_partials = {}
            for ofs in in_of:
                return_partials[ofs] = {}
                for wrts in in_wrt:

                    if self.partial_properties[ofs][wrts]['type'] == 'std':
                        return_partials[ofs][wrts] = partials[ofs][wrts]
                        continue
                    else:
                        ptype = self.partial_properties[ofs][wrts]['type']
                        pp = self.partial_properties[ofs][wrts]

                    if ptype == 'row_col':
                        p = partials[ofs][wrts]
                        pp['val_TM'].data = p
                        return_partials[ofs][wrts] = pp['val_TM']
                        # return_partials[ofs][wrts] = sp.csc_matrix((p, (pp['rows'],pp['cols'])))
                        # return_partials[ofs][wrts] = sp.coo_matrix((p, (pp['rows'],pp['cols']))).tolil()
                        continue

                    elif ptype == 'row_col_val':
                        return_partials[ofs][wrts] = pp['val_TM']
                        continue

                    elif ptype == 'empty':
                        return_partials[ofs][wrts] = pp['val_TM']
                        continue

                    elif ptype == 'cs_uc':
                        cs_uc_run_list.append((wrts, ofs))

                        if wrts not in complex_wrt_dict:
                            complex_wrt_dict[wrts] = []
                        complex_wrt_dict[wrts].append(ofs)
                        continue

                    elif ptype == 'sparse':
                        return_partials[ofs][wrts] = partials[ofs][wrts]
                        continue

        elif approach == 'SB':
            return_partials = {}
            for ofs in in_of:
                return_partials[ofs] = {}
                for wrts in in_wrt:

                    if self.partial_properties[ofs][wrts]['type'] == 'std':
                        return_partials[ofs][wrts] = partials[ofs][wrts]
                        continue
                    else:
                        ptype = self.partial_properties[ofs][wrts]['type']

                    if ptype == 'row_col':
                        return_partials[ofs][wrts] = partials[ofs][wrts]
                        continue

                    if ptype == 'row_col_val':
                        return_partials[ofs][wrts] = self.partial_properties[ofs][wrts]['val']
                        continue

                    if ptype == 'empty':
                        return_partials[ofs][wrts] = None
                        continue

                    elif ptype == 'cs_uc':
                        cs_uc_run_list.append((wrts, ofs))

                        if wrts not in complex_wrt_dict:
                            complex_wrt_dict[wrts] = []
                        complex_wrt_dict[wrts].append(ofs)

                        continue

                    elif ptype == 'sparse':
                        return_partials[ofs][wrts] = partials[ofs][wrts].toarray()
                        continue

        # For all values that need complex step: calculate them
        # if cs_uc_run_list:
        #     dt = self.dt
        #     storage_dict = {}
        #     cs_uc_outs = []
        #     cs_uc_ins = []
        #     print('complex step:')
        #     for (wrts, of) in cs_uc_run_list:
        #         if wrts in cs_uc_ins:
        #             continue
        #         cs_uc_outs.append(of)
        #         cs_uc_ins.append(wrts)

        #         # Record values to set back to initial later
        #         storage_dict[wrts] = self.input_vals[wrts]

        #         # now setting delta i:
        #         wrts_di = np.copy(self.input_vals[wrts])
        #         wrts_di = wrts_di.astype(np.complex)
        #         wrts_di += 0+dt*1j

        #         # setting inputs to perturbed inputs
        #         self.input_vals[wrts] = wrts_di
        #         print('wrts:', wrts)

        #     P = self.run_model({}, cs_uc_outs)
        #     print(cs_uc_outs)

        #     for (wrts, of) in cs_uc_run_list:

        #         # Return ONLY the perterbed value P, not the derivative!!!!!
        #         diagonals = np.imag(P[of].flatten())/self.dt
        #         return_partials[of][wrts] = np.diagflat(diagonals)
        #         # Record values to set back to initial later
        #         self.input_vals[wrts] = storage_dict[wrts]

        if complex_wrt_dict:
            dt = self.dt
            storage_dict = {}
            for wrts in complex_wrt_dict:
                storage_dict[wrts] = self.input_vals[wrts]
                wrts_di = np.copy(self.input_vals[wrts])
                wrts_di = wrts_di.astype(np.complex)
                wrts_di += 0+dt*1j
                self.input_vals[wrts] = wrts_di

                P = self.run_model({}, complex_wrt_dict[wrts])

                for of in complex_wrt_dict[wrts]:
                    diagonals = np.imag(P[of].flatten())/self.dt
                    return_partials[of][wrts] = np.diagflat(diagonals)
                # Record values to set back to initial later
                self.input_vals[wrts] = np.copy(storage_dict[wrts])
        return return_partials

    def set_vars(self, vars):
        # option to set variables
        for key in vars:
            self.input_vals[key] = vars[key]

    def add_input(self, inputs, shape=1):
        self.input_dict[inputs] = {}
        self.input_dict[inputs]['shape'] = shape
        self.input_vals[inputs] = None

    def add_output(self, output, shape=1):
        self.output_dict[output] = {}
        self.output_dict[output]['shape'] = shape
        self.output_vals[output] = None

    def declare_partial_properties(self, of, wrt, rows=None, cols=None, vals=None, complex_step_directional=False, empty=False, sparse=False, standard=True):
        """
        Parameters:
        -----------
            of: str
                variable name corresponding to d{of}/d{wrt}

            wrt: str
                variable name corresponding to d{of}/d{wrt}

            rows: iterable
                Used if fixed row/col indices for sparse Jacobian. 
                If used, cols must have same shape.

            cols: iterable
                Used if fixed row/col indices for sparse Jacobian. 
                If used, rows must have same shape.

            vals: iterable
                Used if fixed row/col indices for sparse Jacobian.
                If used, rows and cols must have same shape.
                Can be left blank and filled in 'self.compute_partials' method

            complex_step_directional: bool
                applies complex step to all state inputs at once.

            empty: bool
                Set true if empty

            sparse: bool
                Set true if Jacobian is a scipy sparse matrix in 'self.compute_partials' method

            standard: bool
                Set true if Jacobian is dense and computed in 'compute_partials'
        """

        # Create list of ofs and list of wrts
        if type(of) is not type([]):
            if of == '*':
                of_list = []
                for output in self.output_dict:
                    of_list.append(output)
            else:
                of_list = [of]
        else:
            of_list = of

        if type(wrt) is not type([]):
            if wrt == '*':
                wrt_list = []
                for input in self.input_dict:
                    wrt_list.append(input)
            else:
                wrt_list = [wrt]
        else:
            wrt_list = wrt

        # Set Jacobian properties
        for of in of_list:
            for wrt in wrt_list:
                if of not in self.partial_properties:
                    self.partial_properties[of] = {}
                self.partial_properties[of][wrt] = {}

                if isinstance(rows, np.ndarray) and isinstance(cols, np.ndarray) and not isinstance(vals, np.ndarray):
                    self.partial_properties[of][wrt]['type'] = 'row_col'
                    self.partial_properties[of][wrt]['rows'] = rows
                    self.partial_properties[of][wrt]['cols'] = cols
                    self.partial_properties[of][wrt]['val_TM'] = sp.csc_matrix(
                        (np.ones(len(cols)), (rows, cols)))
                    continue

                if isinstance(rows, np.ndarray) and isinstance(cols, np.ndarray) and isinstance(vals, np.ndarray):
                    self.partial_properties[of][wrt]['type'] = 'row_col_val'
                    self.partial_properties[of][wrt]['rows'] = rows
                    self.partial_properties[of][wrt]['cols'] = cols
                    self.partial_properties[of][wrt]['val'] = vals
                    self.partial_properties[of][wrt]['val_TM'] = sp.lil_matrix(
                        sp.coo_matrix((vals, (rows, cols))))
                    continue

                if isinstance(empty, bool):
                    if empty == True:
                        self.partial_properties[of][wrt]['type'] = 'empty'

                        num_cols = np.prod(self.input_dict[wrt]['shape'])
                        num_rows = np.prod(self.output_dict[of]['shape'])
                        self.partial_properties[of][wrt]['val_TM'] = sp.lil_matrix(
                            (num_rows, num_cols))
                        continue

                if isinstance(complex_step_directional, bool):
                    if complex_step_directional == True:
                        self.partial_properties[of][wrt]['type'] = 'cs_uc'
                        continue

                if isinstance(sparse, bool):
                    if sparse == True:
                        self.partial_properties[of][wrt]['type'] = 'sparse'
                        continue

                if isinstance(standard, bool):
                    if standard == True:
                        self.partial_properties[of][wrt]['type'] = 'std'
                        continue
        return

    def compute_partials(self, inputs, partials):
        pass

    def compute(self, inputs, outputs):
        pass

    def setup(self):
        pass

    def get_recorder_data(self, var_names):
        save_dict = {}
        for var_name in var_names:
            if var_name == 'number f calls':
                save_dict[var_name] = self.num_f_calls
            elif var_name == 'number vectorized f calls':
                save_dict[var_name] = self.num_vectorized_f_calls
            elif var_name == 'number df calls':
                save_dict[var_name] = self.num_df_calls
            elif var_name == 'number vectorized df calls':
                save_dict[var_name] = self.num_vectorized_df_calls
            else:
                raise KeyError(f'could not find variable {var_name} to save in ozone client')
        return save_dict
