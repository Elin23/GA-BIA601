import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List

class LassoAlgorithm:

    @staticmethod
    def LassoOptimize(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:

        start_time = time.time()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
         
        C_values: List[float] = [1.0, 0.1, 0.05, 0.01, 0.005, 0.001]  
        
        best_accuracy = 0.0 
        best_C = None       
        best_coefficients = None 
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
        
        for C_val in C_values:  
            model = LogisticRegression(penalty='l1', C=C_val, solver='liblinear', max_iter=500, random_state=42) 
            
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy') 
            current_accuracy = np.mean(scores) 
            
            if current_accuracy >= best_accuracy: 
                best_accuracy = current_accuracy
                best_C = C_val
                
                best_model = LogisticRegression(penalty='l1', C=best_C, solver='liblinear', max_iter=500, random_state=42)
                best_model.fit(X_scaled, y)
                best_coefficients = best_model.coef_[0]

        end_time = time.time()
        elapsed_time = end_time - start_time  

        selected_indices = [i for i, coef in enumerate(best_coefficients) if abs(coef) > 1e-4]
        
        return {
            "method": "Lasso Regression (L1 Penalty)",
            "optimal_C_value": float(best_C),
            "selected_features_indices": selected_indices,
            "num_selected_features": len(selected_indices),
            "accuracy": float(best_accuracy),
            "elapsed_time_seconds": elapsed_time
        }