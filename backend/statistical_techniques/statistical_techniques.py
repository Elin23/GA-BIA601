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
        and evaluate accuracy using Linear Regression + 5-fold CV
        """
        start_time = time.time()

        correlations = []
        for i in range(X.shape[1]):
            r, _ = pearsonr(X[:, i], y)
            if np.isnan(r):
                r = 0 
            correlations.append(abs(r))

        
        selected_indices = np.argsort(correlations)[-top_k:][::-1].tolist()

    
        X_selected = X[:, selected_indices]
        scaler = StandardScaler()
        X_selected_scaled = scaler.fit_transform(X_selected)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        model = LinearRegression()
        scores = cross_val_score(model, X_selected_scaled, y, cv=cv, scoring='r2')
        accuracy = float(np.mean(scores))

        elapsed_time = time.time() - start_time

        return {
            "selected_features_indices": selected_indices,
            "num_selected_features": len(selected_indices),
            "accuracy": accuracy,
            "elapsed_time_seconds": elapsed_time
        }
