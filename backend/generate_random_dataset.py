from sklearn.datasets import make_classification
def generate_dataset():
    X, y = make_classification(
    n_samples=500,
    n_features=40,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42
    )
    result = X,y
    return result
