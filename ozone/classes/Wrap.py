
class Wrap(object):
    # Wrap(ODESystem)
    def __init__(self, system, backend):
        self.system = system
        self.system_type = 'OM'

        self.backend = backend
        if self.backend not in ['csdl_om',  'python_csdl_backend']:
            raise ValueError(f'backend must be \'csdl_om\' or \'python_csdl_backend\'')

    def create(self, num_param, type, parameters=None):
        # Creates Problem Object

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
