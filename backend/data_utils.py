import pandas as pd
import numpy as np

def read_dataset(file, target):
    data = pd.read_csv(file)

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    X = data.drop(columns=[target])
    y = data[target]

    non_numeric_cols = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
    if non_numeric_cols:
        raise ValueError(
            f"The following columns contain non-numeric data and must be converted or removed before analysis: "
            f"{', '.join(non_numeric_cols)}"
        )

    X = X.values

    if np.issubdtype(y.dtype, np.floating):
        median_value = y.median()
        y = (y > median_value).astype(int)

  
    elif y.dtype == object or y.dtype == str:
        raise ValueError(
            "The target column contains non-numeric values. Please convert it to numeric or categorical codes before uploading."
        )

    return X, y
