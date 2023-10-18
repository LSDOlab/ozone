import numpy as np
class Wrap(object):
    # Wrap(ODESystem)
    def __init__(self, system, sim_args,  name):
        self.system = system
        self.system_type = 'OM'
        self.name = name
        self.sim_args = sim_args

        if 'name' not in self.sim_args:
            self.sim_args['name'] = self.name

        self.backend = 'python_csdl_backend'

        self.num_f_calls = 0
        self.num_vectorized_f_calls = 0
        self.num_df_calls = 0
        self.num_vectorized_df_calls = 0

        self.recorder = None

        # Tracker to see if model values are different than current values
        # If not, we do not need to actually re-run model when run_model is called
        self.needs_to_run = True

    def create(self, num_param, type, parameters=None):
        # Creates Problem Object
        self.num_nodes = num_param

        import python_csdl_backend
        if parameters is None:
            sim = python_csdl_backend.Simulator(self.system(num_nodes=num_param), **self.sim_args)
        else:
            sim = python_csdl_backend.Simulator(self.system(num_nodes=num_param, **parameters),  **self.sim_args)

        self.problem = sim

    def run_model(self, input_dict, output_vals):
        # Runs model. Can also set variables if needed
        self.num_vectorized_f_calls += 1
        self.num_f_calls += self.num_nodes

        if self.recorder:
            save_dict = self.get_recorder_data(self.recorder.dash_instance.vars['ozone']['var_names'])
            self.recorder.record(save_dict, 'ozone')

        for key, value in input_dict.items():
            if self.needs_to_run == False:
                if not np.array_equal(self.problem[key], value):
                    self.needs_to_run = True

            self.problem[key] = value

        if self.needs_to_run:
            self.problem.run()
            self.needs_to_run = False
            # print('RAN PROBLEM')
        else:
            # print('AVOIDED RUN')
            pass

        outputs = {}
        for key in output_vals:
            outputs[key] = self.problem[key]
        return outputs

    def compute_total_derivatives(self, in_of, in_wrt, approach='TM', vjp = None):

        self.num_df_calls += self.num_nodes
        self.num_vectorized_df_calls += 1
        if self.recorder:
            save_dict = self.get_recorder_data(self.recorder.dash_instance.vars['ozone']['var_names'])
            self.recorder.record(save_dict, 'ozone')

        # Computes Derivatives
        if vjp is None:
            return self.problem.compute_totals(of=in_of, wrt=in_wrt, return_format='dict')
        else:
            vjps_edited = self.problem.compute_vector_jacobian_product(of_vectors=vjp, wrt=in_wrt,return_format='dict')

            used_wrts = set()
            return_dict = {}
            for of in in_of:
                if of not in vjps_edited:
                    return_dict[of] = {}

                for wrt in in_wrt:
                    if wrt not in return_dict[of]:
                        if wrt not in used_wrts:
                            used_wrts.add(wrt)
                            return_dict[of][wrt] = vjps_edited[wrt]
                        else:
                            return_dict[of][wrt] = np.zeros((vjps_edited[wrt].shape))
                    else:
                        return_dict[of][wrt] = np.zeros((vjps_edited[wrt].shape))

            return return_dict

    def set_vars(self, vars):
        # option to set variables
        for key in vars:
            if self.needs_to_run == False:
                if not np.array_equal(self.problem[key], vars[key]):
                    self.needs_to_run = True

            self.problem[key] = vars[key]
            # print('SET VARS:', key, self.problem[key], vars[key])

    def check_totals(self, in_of, in_wrt):
        if self.backend == 'csdl_om':
            self.problem.run_model()
        else:
            self.problem.run()
        self.problem.check_totals(of=in_of, wrt=in_wrt, compact_print=True)

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
