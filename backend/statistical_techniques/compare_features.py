import numpy as np
from sklearn.datasets import make_classification
import json
from statistical_techniques import StatsFeatureSelection  

def generate_dataset():
    X, y = make_classification(
        n_samples=500,
        n_features=40,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return X, y

def main():

    X, y = generate_dataset()

    print("Running Correlation-based Feature Selection...")
    correlation_results = StatsFeatureSelection.correlation_selection(X, y, top_k=10)


    print(json.dumps(correlation_results, indent=4))

if __name__ == "__main__":
    main()
