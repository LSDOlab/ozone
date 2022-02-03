import numpy as np

def get_integration_method(string):

    if string == 'RK4':
        A = np.array(
            [[0., 0., 0., 0.],
             [0.5, 0., 0., 0.],
             [0., 0.5, 0., 0.],
             [0., 0., 1., 0.]])
        B = np.array([[1 / 6, 1 / 3, 1 / 3, 1 / 6]])
        U = np.array([[1.], [1.], [1.], [1.]])
        V = np.array([[1.]])
    elif string == 'RK3':
        A = np.array(
            [[0., 0., 0.],
            [1 / 2, 0., 0.],
            [-1., 2., 0.]])
        B = np.array(
            [[1 / 6, 4 / 6, 1 / 6]])
        U = np.array([[1.], [1.], [1.]])
        V = np.array([[1.]])
    elif string == 'Trapezoidal':
        A = np.array(
            [[0., 0.],
           [1 / 2, 1 / 2]])
        B = np.array([[1 / 2, 1 / 2]])
        U = np.array([[1.], [1.]])
        V = np.array([[1.]])
    else:
        print('Integration method does not exist')
        return None

    return A,B,U,V

    