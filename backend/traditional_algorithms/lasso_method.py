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
        
        # Initialize the scaler object and scale the features (X) to have zero mean and unit variance.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # C values for Grid Search: C is the inverse of the regularization strength.     
        C_values: List[float] = [1.0, 0.1, 0.05, 0.01, 0.005, 0.001]  #define the set of C values to test
        
        best_accuracy = 0.0 #variable to track the highest achieved accuracy
        best_C = None       #variable to store the C value that yields best_accuracy.
        best_coefficients = None #variable to store the feature coefficients of the best model.
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #CV splitter ensuring class balance (stratify=y)
        
        for C_val in C_values:  #loop through each defined C value.
            model = LogisticRegression(penalty='l1', C=C_val, solver='liblinear', max_iter=500, random_state=42) #instantiate the model for the current C.
            
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy') #test the model 5 times (5-fold CV) and get the scores and calculate the average score
            current_accuracy = np.mean(scores) 
            
            #If the current score is better or equal to the best score found so far update C and accuracy
            if current_accuracy >= best_accuracy: 
                best_accuracy = current_accuracy
                best_C = C_val
                
                # retrain the optimal model and extract feature weights
                best_model = LogisticRegression(penalty='l1', C=best_C, solver='liblinear', max_iter=500, random_state=42)
                best_model.fit(X_scaled, y)
                best_coefficients = best_model.coef_[0]

        end_time = time.time()
        elapsed_time = end_time - start_time  

        # determine Selected Features (Features with coefficients close to zero (below 1e-4) are eliminated)
        selected_indices = [i for i, coef in enumerate(best_coefficients) if abs(coef) > 1e-4]
        
        return {
            "method": "Lasso Regression (L1 Penalty)",
            "optimal_C_value": float(best_C),
            "selected_features_indices": selected_indices,
            "num_selected_features": len(selected_indices),
            "accuracy": float(best_accuracy),
            "elapsed_time_seconds": elapsed_time
        }
