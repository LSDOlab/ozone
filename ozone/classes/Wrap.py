import csdl_om


class Wrap(object):
    # Wrap(ODESystem)
    def __init__(self, system):
        self.system = system
        self.system_type = 'OM'

    def create(self, num_param, type, parameters=None):
        # Creates Problem Object
        if type == 'O':
            if parameters is None:
                sim = csdl_om.Simulator(self.system(num_nodes=num_param))
            else:
                sim = csdl_om.Simulator(self.system(num_nodes=num_param, **parameters))
        elif type == 'P':
            if parameters is None:
                sim = csdl_om.Simulator(self.system(num_nodes=num_param))
            else:
                sim = csdl_om.Simulator(self.system(num_nodes=num_param, **parameters))

        self.problem = sim.prob
        # import openmdao.api as OM
        # OM.n2(self.problem)
        # self.problem.setup()

    def run_model(self, input_dict, output_vals):
        # Runs model. Can also set variables if needed

        for key, value in input_dict.items():
            self.problem[key] = value
        self.problem.run_model()

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
            self.problem.set_val(key, vars[key])
            # print('SET VARS:', key, self.problem[key], vars[key])

    def check_totals(self, in_of, in_wrt):
        self.problem.run_model()
        self.problem.check_totals(of=in_of, wrt=in_wrt, compact_print=True)
