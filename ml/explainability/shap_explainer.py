import os
import joblib
import shap
import logging
import numpy as np
import json

logging.basicConfig(level=logging.INFO, format="%(message)s")

class SHAPExplainer:
    def __init__(self, models_dir="models/"):
        self.models_dir = models_dir
        
        # Load Preprocessor
        self.preprocessor_data = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
        self.pipeline = self.preprocessor_data['pipeline']
        self.feature_names = self.preprocessor_data['features']
        
        # Load Model (XGBoost is preferred for SHAP TreeExplainer)
        self.model = joblib.load(os.path.join(models_dir, "xgboost.joblib"))
        
        # Initialize Explainer
        logging.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_instance(self, df, top_k=5):
        """
        Explain a single instance or batch of instances.
        Returns a list of explanation dictionaries.
        """
        # Preprocess
        X_processed = self.pipeline.transform(df)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_processed)
        
        # If multi-class, shap_values is a list of arrays (one per class).
        # We need to find the predicted class to get the relevant explanation.
        preds = self.model.predict(X_processed)
        
        explanations = []
        for i in range(len(df)):
            pred_class_idx = preds[i]
            
            if isinstance(shap_values, list):
                # SHAP returns a list of arrays (one per class)
                instance_shap = shap_values[pred_class_idx][i]
            elif len(shap_values.shape) == 3:
                # Shape (n_samples, n_features, n_classes)
                instance_shap = shap_values[i, :, pred_class_idx]
            else:
                instance_shap = shap_values[i]
                
            # Combine feature names, actual values, and SHAP contributions
            original_values = df.iloc[i].to_dict()
            processed_values = X_processed[i]
            
            contributions = []
            for j, feat_name in enumerate(self.feature_names):
                val = instance_shap[j]
                # If val is still an array, take the first element (fallback)
                if hasattr(val, '__len__'):
                    val = val[0]
                contributions.append({
                    "feature": feat_name,
                    "original_value": original_values.get(feat_name, None),
                    "processed_value": float(processed_values[j]),
                    "shap_value": float(val)
                })
                
            # Sort by absolute SHAP value (impact magnitude)
            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            
            explanations.append({
                "predicted_class_idx": int(pred_class_idx),
                "top_features": contributions[:top_k]
            })
            
        return explanations

if __name__ == "__main__":
    import pandas as pd
    explainer = SHAPExplainer()
    df = pd.read_csv("data/raw/dataset.csv", nrows=1)
    res = explainer.explain_instance(df)
    logging.info(f"SHAP Explanation for first row: {json.dumps(res, indent=2)}")
