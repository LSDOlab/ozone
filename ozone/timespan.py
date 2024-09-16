import csdl_alpha as csdl
from ozone.utils.general import variablize, check_if_int
import numpy as np

def check_if_scalar(value:csdl.Variable, argname:str)->csdl.Variable:
    if not value.size == 1:
        raise ValueError(f'expected scalar for argument \'{argname}\' but got shape {value.shape}')
    if value.shape != (1,):
        value = value.reshape((1,))
    return value

def check_if_vector(value:csdl.Variable, argname:str)->csdl.Variable:
    if len(value.shape) == 1:
        return value
    else:
        raise ValueError(f'expected 1D vector for argument \'{argname}\' but got shape {value.shape}')

class TimeSpan(object):
    def __init__(
            self,
        ) -> None:
        self.step_vector:csdl.Variable = None
        self.time_vector:csdl.Variable = None

    def finalize(self):
        time_vector = self.time_vector
        step_vector = self.step_vector
        num_times = step_vector.size+1
        if num_times == 1:
            raise ValueError('Atleast two time points are required to define a time span.')
        if step_vector.shape != (num_times-1,):
            raise ValueError(f'step_vector has shape {step_vector.shape} but expected shape {(num_times-1,)}')
        return time_vector, step_vector, num_times

class TimeSeries(object):
    def __init__(
            self,
            time_vector:csdl.Variable,
        ) -> None:

        time_vector = variablize(time_vector)
        time_vector = check_if_vector(time_vector, 'time_vector')
        self.time_vector:csdl.Variable = time_vector
        self.step_vector:csdl.Variable = None

class StartEnd(TimeSpan):
    def __init__(
            self,
            start:csdl.Variable,
            end:csdl.Variable,
            num_steps:int,
        ) -> None:
        super().__init__()
        start = variablize(start)
        end = variablize(end)
        start = check_if_scalar(start, 'start')
        end = check_if_scalar(end, 'end')
        check_if_int(num_steps, 'num_steps')

        self.step_vector = csdl.linspace(start, end, num_steps) - start
        # self.time_vector = csdl.cumsum(self.step_vector)+start

class StepVector(TimeSpan):
    def __init__(
            self,
            start:csdl.Variable,
            step_vector:csdl.Variable,
        ) -> None:
        super().__init__()
        start = variablize(start)
        start = check_if_scalar(start, 'start')
        step_vector = variablize(step_vector)
        step_vector = check_if_vector(step_vector, 'step_vector')

        self.step_vector = step_vector
        # self.time_vector = csdl.cumsum(self.step_vector)+start

# class StartStep(TimeSpan):
#     def __init__(
#             self,
#             start:csdl.Variable,
#             step_size:csdl.Variable,
#             num_steps:int,
#         ) -> None:
#         super().__init__()
#         start = variablize(start)
#         start = check_if_scalar(start, 'start')
#         step_size = variablize(step_size)
#         if step_size.size == 1:
#             step_vector = np.ones(num_steps)*step_size
#         step_vector = variablize(step_vector)

#         check_if_int(num_steps, 'num_steps')

#         self.step_vector = step_vector
#         # self.time_vector = csdl.cumsum(self.step_vector)+start