import pandas as pd
from sklearn.preprocessing import LabelEncoder
<<<<<<< HEAD
import os
def read_dataset(file,target):
    data=pd.read_csv(file)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    
    x=data.drop(columns=[target])
    y=data[target]
    
    


    label_encoders = {}
    for col in x.columns:
        if x[col].dtype == 'object' or x[col].dtype.name == 'category': 
            le = LabelEncoder()
            x[col] = le.fit_transform(x[col].astype(str))
            label_encoders[col] = le
    if y.dtype == 'object':
        y_le = LabelEncoder()
        y = y_le.fit_transform(y.astype(str))
        label_encoders[target] = y_le
    output_path = os.path.join(os.path.dirname(__file__), "dataset_processed.txt")
    with open(output_path, "w") as f:
        f.write("X values:\n")
        for row in x.values:
            f.write(",".join(map(str, row)) + "\n")
        f.write("\nY values:\n")
        f.write(",".join(map(str, y)))
    return x.values,y
=======
import category_encoders as ce

def read_dataset(file, target):
    data = pd.read_csv(file)

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    X = data.drop(columns=[target])
    y = data[target]

    categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
    if categorical_cols:
        encoder = ce.TargetEncoder(cols=categorical_cols)
        X = encoder.fit_transform(X, y)  

    if y.dtype == 'object' or y.dtype.name == 'category':
        y_le = LabelEncoder()
        y = y_le.fit_transform(y)

    return X.values, y
>>>>>>> 1dbf00b8ca6258c8c6a239432ce27870874dacd3
