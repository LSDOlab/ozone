
import numpy as np


def lin_interp(params, C, numtimes, nn_shape):
    """
    This functions takes in parameters for a timestep and splits them up with linear interpolation
    https://en.wikipedia.org/wiki/Linear_interpolation
    """

    # What to return: an "array" with shape (numtimes,n, ...param_shape...)
    nodal_param = []

    # For each timestep perform linear interpolation
    for t in range(numtimes):
        # Create parameter for each timestep
        temp = np.zeros(nn_shape)

        # Current param value
        param_now = params[t]
        param_next = params[t+1]

        # # Next param value
        # if t != numtimes-1:
        #     param_next = params[t+1]
        # else:
        #     param_next = params[t]

        # Linear interpolation
        for stages, c_step in enumerate(C):
            temp[stages] = param_now * (1.0 - c_step) + param_next*(c_step)

        # # OLD: WITHOUT Linear interpolation
        # for stages, c_step in enumerate(C):
        #     temp[stages] = param_now

        # Store
        nodal_param.append(temp)
    nodal_param = np.array(nodal_param)

    return nodal_param
