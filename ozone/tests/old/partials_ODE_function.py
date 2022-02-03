
from ozone.tests.functions.create_ODE_function import create_prob
import numpy as np


def partials_prob(approach_list, num_list, h, method_list, input_list, ODEProblemTest, of=None, wrt=None):

    # Solve ODE for each of these methods
    output_values = {}
    for i, num in enumerate(num_list):
        output_values[str(num)] = []
        inputs_num = input_list[i]
        for approach in approach_list:
            for method in method_list:
                prob = create_prob(num, h, method, approach,
                                   inputs_num, ODEProblemTest)
                prob.run_model()
                p_out = prob.check_totals(of=of, wrt=wrt, compact_print=True)
                output_values[str(num)].append(p_out)
    return output_values
