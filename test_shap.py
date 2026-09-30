import pickle
import shap
import numpy as np
with open('models/saved/xgboost_seed1.pkl', 'rb') as f:
    xgb = pickle.load(f)
explainer = shap.TreeExplainer(xgb)
X = np.random.randn(10, 7)
sv = explainer.shap_values(X)
print(type(sv), sv.shape if isinstance(sv, np.ndarray) else [s.shape for s in sv])
