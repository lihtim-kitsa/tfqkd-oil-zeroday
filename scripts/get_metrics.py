import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_parquet('data/dataset_v1.parquet')
X = df[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = joblib.load('checkpoints/scaler.pkl')
X_test = scaler.transform(X_test)

for model_name in ['xgboost', 'randomforest', 'svm']:
    print(f"--- {model_name} ---")
    try:
        model = joblib.load(f'checkpoints/{model_name}.pkl')
        preds = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(classification_report(y_test, preds, target_names=['Nominal', 'FIM', 'TWIRL']))
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
