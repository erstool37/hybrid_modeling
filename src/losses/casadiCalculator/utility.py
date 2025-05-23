import casadi as ca
from pathlib import Path

# scale X -> [0,1]
def zero_one_scale(var, var_min, var_max):
    return (var - var_min) / (var_max - var_min)

# descale [0,1] -> X
def zero_one_descale(var, var_min, var_max):
    return (var_max - var_min) * var + var_min

# make directory using pathlib
def make_directory(*args):
    directory = Path.cwd()
    for arg in args:
        directory = directory / arg
        directory.mkdir(exist_ok=True)
    print('Directory:', directory, 'is made')
    return directory