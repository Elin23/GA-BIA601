import random
from sklearn.datasets import make_classification
def generate_dataset():
    seed = random.randint(0, 10000)
    X, y = make_classification(
    n_samples=500,
    n_features=40,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=seed
    )
    result = X,y
    return result
