import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
