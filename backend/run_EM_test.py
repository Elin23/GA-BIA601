import numpy as np
from embedded_methods import EmbeddedMethod
import pandas as pd
import json

# Load the data
df = pd.read_csv("dataset.csv")

# Prepare the target variable
y = df["Model"].values

# Prepare features: drop the target and encode categoricals
X = df.drop(columns=["Model"])
X = pd.get_dummies(X)            # One-hot encode all categoricals
X = X.values                     # Get numpy array

# Run the embedded method
result = EmbeddedMethod.run(X, y)

print(json.dumps(result, indent=4))