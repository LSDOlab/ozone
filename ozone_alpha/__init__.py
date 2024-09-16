__version__ = '0.0.0-a.0'

# Main ODE interface
from ozone_alpha.odeproblem import ODEProblem

# helper functions to define time spans
import ozone_alpha.timespan as timespans

# approaches and methods
import ozone_alpha.approaches as approaches
import ozone_alpha.methods as methods

# for typehinting (Not necessary for the user to import)
from ozone_alpha.func_wrapper import FuncVars
from ozone_alpha.collections import IntegratorOutputs