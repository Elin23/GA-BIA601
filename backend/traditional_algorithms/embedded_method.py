import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

      
class EmbeddedMethod:
    @staticmethod
    def run(X, y, top_k=20):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        idx_topk = np.argsort(importances)[-top_k:][::-1]
        model_rf_subset = LogisticRegression(max_iter=1000, random_state=42)
        model_rf_subset.fit(X_train[:, idx_topk], y_train)
        preds_rf = model_rf_subset.predict(X_val[:, idx_topk])
        acc_rf = accuracy_score(y_val, preds_rf)
        baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_preds = baseline_model.predict(X_val)
        baseline_acc = accuracy_score(y_val, baseline_preds)
        return {
            'selected_feature_indices': idx_topk.tolist(),
            'accuracy_rf': float(acc_rf),
            'accuracy_baseline': float(baseline_acc),
            'num_selected_features': len(idx_topk),
            'total_features': X.shape[1],
            'feature_importances': importances[idx_topk].tolist()
        }
