
class Wrap(object):
    # Wrap(ODESystem)
    def __init__(self, system, backend):
        self.system = system
        self.system_type = 'OM'

        self.backend = backend
        if self.backend not in ['csdl_om',  'python_csdl_backend']:
            raise ValueError(f'backend must be \'csdl_om\' or \'python_csdl_backend\'')

        self.num_f_calls = 0
        self.num_vectorized_f_calls = 0
        self.num_df_calls = 0
        self.num_vectorized_df_calls = 0

        self.recorder = None

    def create(self, num_param, type, parameters=None):
        # Creates Problem Object
        self.num_nodes = num_param

        if self.backend == 'csdl_om':
            import csdl_om
            from csdl import GraphRepresentation
            if parameters is None:
                sim = csdl_om.Simulator(GraphRepresentation(self.system(num_nodes=num_param)))
            else:
                sim = csdl_om.Simulator(GraphRepresentation(self.system(num_nodes=num_param, **parameters)))

            self.problem = sim.executable
        elif self.backend == 'python_csdl_backend':
            import python_csdl_backend
            if parameters is None:
                sim = python_csdl_backend.Simulator(self.system(num_nodes=num_param))
            else:
                sim = python_csdl_backend.Simulator(self.system(num_nodes=num_param, **parameters))

            self.problem = sim

    def run_model(self, input_dict, output_vals):
        # Runs model. Can also set variables if needed
        self.num_vectorized_f_calls += 1
        self.num_f_calls += self.num_nodes

        if self.recorder:
            save_dict = self.get_recorder_data(self.recorder.dash_instance.vars['ozone']['var_names'])
            self.recorder.record(save_dict, 'ozone')

        for key, value in input_dict.items():
            self.problem[key] = value

        if self.backend == 'csdl_om':
            self.problem.run_model()
        else:
            self.problem.run()

        outputs = {}
        for key in output_vals:
            outputs[key] = self.problem[key]
        return outputs

    def compute_total_derivatives(self, in_of, in_wrt, approach='TM'):

        self.num_df_calls += self.num_nodes
        self.num_vectorized_df_calls += 1
        if self.recorder:
            save_dict = self.get_recorder_data(self.recorder.dash_instance.vars['ozone']['var_names'])
            self.recorder.record(save_dict, 'ozone')

        # Computes Derivatives
        if approach == 'TM':
            return self.problem.compute_totals(of=in_of, wrt=in_wrt, return_format='dict')
        elif approach == 'SB':
            return self.problem.compute_totals(of=in_of, wrt=in_wrt, return_format='dict')

    def set_vars(self, vars):
        # option to set variables
        for key in vars:
            # self.problem.set_val(key, vars[key])
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
