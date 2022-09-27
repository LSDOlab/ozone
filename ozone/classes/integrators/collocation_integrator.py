from ozone.classes.integrators.vectorized_integrator import VectorBased
from ozone.classes.integrators.vectorized.MainGroup import VectorBasedGroup


class Collocation(VectorBased):

    def get_solver_model(self):

        # return SolverBased group
        component = VectorBasedGroup(solution_type = 'collocation')
        component.add_ODEProb(self)

        return component
