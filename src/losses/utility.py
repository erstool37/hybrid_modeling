from pathlib import Path

# scale X -> [0,1]
def zero_one_scale(var, var_min, var_max):
    scaled_var = (var - var_min) / (var_max - var_min)
    return scaled_var


# descale [0,1] -> X
def zero_one_descale(var, var_min, var_max):
    descaled_var = (var_max - var_min) * var + var_min
    return descaled_var


def make_directory(*args):
    directory = Path.cwd()
    
    for arg in args:
        directory = Path.joinpath(directory, arg)
        Path(directory).mkdir(exist_ok=True)
        
    print('Directory:', directory, 'is made')
    return directory
        