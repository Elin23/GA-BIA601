import numpy as np
from ga_algorithm import GAAlgorithm
from sklearn.datasets import make_classification
import json

X, y = make_classification(
    n_samples=500,
    n_features=40,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

result = GAAlgorithm.GAOptimize(X, y)

print(json.dumps(result, indent=4))
