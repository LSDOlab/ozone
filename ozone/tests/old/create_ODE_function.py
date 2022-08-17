
# # Create a test by solving an ODE using the different approaches
# def run ODE:
import csdl
import python_csdl_backend


def create_prob(num_t, h, method, approach, inputs, ODEProblemClass):

    # Create Model that ODE is in
    class ODEModel(csdl.Model):
        def define(self):

            num = num_t
            h_initial = h

            # Create given inputs
            for input_name in inputs:
                self.create_input(input_name, inputs[input_name])

            # ODEProblem_instance
            ODEProblem_instance = ODEProblemClass(
                method, approach, num_times=num, display='default', visualization='none')
            self.add(ODEProblem_instance.create_solver_model(), 'subgroup')

            # Output
            out = self.declare_variable('output')
            # self.register_output('out', out)

    # Backend problem that can be ran.
    sim = python_csdl_backend.Simulator(ODEModel(), mode='rev')
    return sim.prob
