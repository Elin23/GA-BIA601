import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

class StatsFeatureSelection:

    @staticmethod
    def correlation_selection(X, y, top_k=10):
        """
        Select top_k features with highest absolute Pearson correlation with target
        """
        start_time = time.time()
        correlations = [abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])]
        selected_indices = np.argsort(correlations)[-top_k:].tolist()

        # evaluate performance
        scaler = StandardScaler()
        X_selected = scaler.fit_transform(X[:, selected_indices])
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        model = LinearRegression()
        scores = cross_val_score(model, X_selected, y, cv=cv, scoring='r2')
        elapsed_time = time.time() - start_time

        return {
            "selected_features_indices": selected_indices,
            "num_selected_features": len(selected_indices),
            "score": float(np.mean(scores)),
            "elapsed_time_seconds": elapsed_time
        }