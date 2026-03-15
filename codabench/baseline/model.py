import pandas as pd
import numpy as np

class Model:
    def __init__(self):
        self.majority_label = None

    def fit(self, X_train, y_train):
        if y_train.empty:
            raise ValueError("Training labels are empty")
        
        self.majority_label = y_train['category'].value_counts().idxmax()


    def predict(self, X_test):
        if self.majority_label is None:
            raise ValueError("Model must be fit before calling predict")
        
        test_rows = np.asarray(X_test)
        
        if test_rows.ndim == 1:
            sample_count = 1 if test_rows.size > 0 else 0
        else:
            sample_count = test_rows.shape[0]
        
        # Create an array of predictions
        return np.full(shape=(sample_count,), fill_value=self.majority_label)