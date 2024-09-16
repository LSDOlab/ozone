
def get_type_string(value):
    return f'\'{type(value).__name__}\''

def check_if_string(value, argname):
    if not isinstance(value, str):
        raise TypeError(f'expected type \'string\' for argument \'{argname}\' but got {get_type_string(value)}')

def check_if_method(value, argname):
    from ozone_alpha.methods import _Method
    if not isinstance(value, _Method):
        raise TypeError(f'expected type \'ozone.methods._Method\' for argument \'{argname}\' but got {get_type_string(value)}')

def check_if_variable(value, argname):
    from csdl_alpha import Variable
    if not isinstance(value, Variable):
        raise TypeError(f'expected type \'Variable\' for argument \'{argname}\' but got {get_type_string(value)}')
    
def check_if_bool(value, argname):
    if not isinstance(value, bool):
        raise TypeError(f'expected type \'bool\' for argument \'{argname}\' but got {get_type_string(value)}')

def check_if_int(value, argname):
    import numpy as np
    if not isinstance(value, (int, np.int32, np.int64)):
        raise TypeError(f'expected type \'int\' for argument \'{argname}\' but got {get_type_string(value)}')

reversed_alphabet = 'zyxwvutsrqponmlkjihgfedcba'
def get_general_expand_string(shape:tuple[int])->str:
    if len(shape) > len(reversed_alphabet):
        raise ValueError(f'Cannot expand shape of length {len(shape)}')
    return reversed_alphabet[:len(shape)]

def variablize(value):
    if isinstance(value, list):
        import numpy as np
        value = np.array(value)
    from csdl_alpha.utils.inputs import variablize
    return variablize(value)