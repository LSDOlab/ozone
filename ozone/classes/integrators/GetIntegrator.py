from ozone.classes.integrators.TimeMarching import TimeMarching
from ozone.classes.integrators.TimeMarchingWithCheckpointing import TimeMarchingWithCheckpointing
from ozone.classes.integrators.solver_based_integrator import SolverBased
from ozone.classes.integrators.collocation_integrator import Collocation


def get_integrator(approach):
    """
    Returns integrator class depending on approach.

    Parameters
    ----------
        approach: str
            Approach specified by user
    """
    return integrator_dict[approach]


# Dictionary of integrator string and appropriate class
integrator_dict = {'time-marching': TimeMarching,
                   'solver-based': SolverBased,
                   'time-marching checkpointing': TimeMarchingWithCheckpointing,
                   'collocation': Collocation,
                   }
