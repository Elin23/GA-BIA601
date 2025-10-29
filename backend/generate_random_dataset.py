import numpy as np
def generate_dataset(num_rows=200, num_cols=40, target_type='binary'):
    X = np.random.rand(num_rows, num_cols)
    if target_type == 'binary':
        y = np.random.randint(0, 2, size=num_rows)
    elif target_type == 'multiclass':
        y = np.random.randint(0, 3, size=num_rows)
    elif target_type == 'continuous':
        y = np.random.rand(num_rows)
    else:
        raise ValueError("target_type must be 'binary', 'multiclass' or 'continuous'")
    return X, y
