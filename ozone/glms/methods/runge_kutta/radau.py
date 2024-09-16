from __future__ import division

import numpy as np

from ozone.glms.methods.runge_kutta.runge_kutta import RungeKutta

_radau_I_coeffs = {
    3: (
        np.array([[1 / 4, -1 / 4],
                  [1 / 4, 5 / 12]]),
        np.array([1 / 4, 3 / 4])
    ),
    5: (
        np.array([[1 / 9, (-1 - np.sqrt(6)) / 18, (-1 + np.sqrt(6)) / 18],
                  [1 / 9, 11 / 45 + 7*np.sqrt(6) / 360, 11 / 45 - 43*np.sqrt(6) / 360],
                  [1 / 9, 11 / 45 + 43*np.sqrt(6) / 360, 11 / 45 - 7*np.sqrt(6) / 360]]),
        np.array([1 / 9, 4 / 9 + np.sqrt(6) / 36, 4 / 9 - np.sqrt(6) / 36])
    )
}

_radau_II_coeffs = {
    3: (
        np.array([[5 / 12, -1 / 12],
                  [3 / 4, 1 / 4]]),
        np.array([3 / 4, 1 / 4])
    ),
    5: (
        np.array([[11 / 45 - 7*np.sqrt(6) / 360, 37 / 225 - 169 * np.sqrt(6) / 1800, -2 / 225 + np.sqrt(6) / 75],
                  [37 / 225 + 169*np.sqrt(6) / 1800, 11 / 45 + 7*np.sqrt(6) / 360, -2 / 225 - np.sqrt(6) / 75],
                  [4 / 9 - np.sqrt(6) / 36, 4 / 9 + np.sqrt(6) / 36, 1 / 9]]),
        np.array([4 / 9 - np.sqrt(6) / 36, 4 / 9 + np.sqrt(6) / 36, 1 / 9])
    )
}


class Radau(RungeKutta):

    def __init__(self, type_, order=5):
        self.order = order

        if type_ == 'I':
            coeffs = _radau_I_coeffs
        elif type_ == 'II':
            coeffs = _radau_II_coeffs

        if order not in coeffs:
            raise ValueError('Radau order must be one of the following: {}'.format(
                sorted(coeffs.keys())
            ))
        A, B = coeffs[order]
        super(Radau, self).__init__(A=A, B=B)
