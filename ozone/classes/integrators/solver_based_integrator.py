from ozone.classes.integrators.vectorized_integrator import VectorBased
from ozone.classes.integrators.vectorized.MainGroup import VectorBasedGroup


class SolverBased(VectorBased):

    def get_solver_model(self):

        # return SolverBased group
        component = VectorBasedGroup(solution_type = 'solver-based')
        component.add_ODEProb(self)

        return component
