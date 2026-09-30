from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_xgb_model():
    """Returns XGBoost configured with 500 trees."""
    return XGBClassifier(n_estimators=500, eval_metric='logloss')

def get_svm_model():
    """Returns SVM with RBF kernel, scaled properly."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True))
    ])

def get_rf_model():
    """Returns Random Forest configured with 500 trees."""
    return RandomForestClassifier(n_estimators=500)

def get_logreg_model():
    """Returns Multinomial Logistic Regression baseline model (softmax)."""
    return LogisticRegression(solver='lbfgs', max_iter=1000)
