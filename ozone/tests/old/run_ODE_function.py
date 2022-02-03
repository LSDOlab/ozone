
from ozone.tests.functions.create_ODE_function import create_prob
import numpy as np


def run_prob(approach_list, num_list, h, method_list, input_list, ODEProblemTest):

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
                output_values[str(num)].append(np.linalg.norm(prob['output']))
    return output_values
