import numpy as np
import shap
import joblib

class ResponseController:
    def __init__(self, model_path='checkpoints/xgboost.pkl'):
        """
        Initializes the Response Controller with the loaded model and SHAP explainer.
        """
        try:
            self.model = joblib.load(model_path)
            self.explainer = shap.TreeExplainer(self.model)
            self.scaler = joblib.load('checkpoints/scaler.pkl')
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            self.model = None
            self.explainer = None
            self.scaler = None
            
        self.feature_names = ['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']
        
    def evaluate(self, feature_vector):
        """
        Evaluates a single sample or batch of samples and assigns a Tier based on the roadmap.
        feature_vector: array-like of shape (n_samples, 5)
        """
        if self.model is None or self.scaler is None:
            return "Model or scaler not initialized properly."
            
        feature_vector = self.scaler.transform(feature_vector)
            
        # Predict probability of attack (class 1 or higher)
        # XGBoost output depends on multiclass vs binary, here we assume predict_proba is available
        probs = self.model.predict_proba(feature_vector)
        
        # If multiclass, probability of attack is 1 - P(Normal), where Normal is class 0
        if probs.shape[1] > 2:
            p_attack = 1 - probs[:, 0]
        else:
            p_attack = probs[:, 1]
            
        results = []
        for i, p in enumerate(p_attack):
            tier, action = self._get_tier(p)
            
            if tier > 0:
                # Generate SHAP explanation
                explanation = self._generate_explanation(feature_vector[i:i+1])
                msg = f"Alert {tier}: {action} (P_attack={p:.2f}) -> Driven by: {explanation}"
            else:
                msg = f"Alert 0: {action} (P_attack={p:.2f})"
                
            results.append(msg)
            
        return results

    def _get_tier(self, p_attack):
        if p_attack < 0.15:
            return 0, "Normal - Standard logging only"
        elif p_attack < 0.50:
            return 1, "Warning - Enhanced monitoring"
        elif p_attack < 0.85:
            return 2, "Suspected - Reduce key rate; flag session"
        else:
            return 3, "Confirmed - Abort session; quarantine reference beam"
            
    def _generate_explanation(self, x):
        shap_values = self.explainer.shap_values(x)
        
        if isinstance(shap_values, list): # Multiclass XGBoost returns list of arrays
            # Use importance for attack classes (skip class 0 which is normal)
            importance = np.zeros(len(self.feature_names))
            for i in range(1, len(shap_values)):
                importance += np.abs(shap_values[i][0])
        else:
            importance = np.abs(shap_values[0])
            
        top_indices = np.argsort(importance)[::-1]
        top_2 = [self.feature_names[top_indices[0]], self.feature_names[top_indices[1]]]
        return f"{top_2[0]} and {top_2[1]}"

if __name__ == "__main__":
    # Test the controller
    print("Testing Response Controller...")
    controller = ResponseController()
    
    # Synthetic samples (mu, sigma2, Psb, delta_phi, qber)
    # 1. Normal (low values)
    # 2. FIM Attack (high mu, sigma2)
    # 3. TWIRL Attack (high Psb, delta_phi)
    
    test_samples = np.array([
        [0.01, 0.001, 0.005, 0.01, 0.02],
        [0.80, 0.450, 0.005, 0.01, 0.07],
        [0.01, 0.001, 0.950, 0.85, 0.06]
    ])
    
    alerts = controller.evaluate(test_samples)
    for alert in alerts:
        print(alert)
