import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})

def evaluate_models():
    # Load drift dataset
    df_drift = pd.read_parquet('data/drift_dataset.parquet')
    X_drift_raw = df_drift[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].values
    
    # Scaled features for baseline models
    scaler = joblib.load('checkpoints/scaler.pkl')
    X_drift_scaled = scaler.transform(X_drift_raw)
    
    base_models = ['xgboost', 'randomforest']
    results = []
    
    for model_name in base_models:
        # Baseline FPR
        path_base = f'checkpoints/{model_name}.pkl'
        if os.path.exists(path_base):
            model_base = joblib.load(path_base)
            drift_preds_base = model_base.predict(X_drift_scaled)
            fp_base = np.sum(drift_preds_base > 0)
            tn_base = np.sum(drift_preds_base == 0)
            fpr_base = fp_base / (fp_base + tn_base) if (fp_base + tn_base) > 0 else 0
        else:
            fpr_base = 0
            
        # Hardened FPR
        path_hard = f'checkpoints/{model_name}_hardened.pkl'
        if os.path.exists(path_hard):
            model_hard = joblib.load(path_hard)
            drift_preds_hard = model_hard.predict(X_drift_raw)
            fp_hard = np.sum(drift_preds_hard > 0)
            tn_hard = np.sum(drift_preds_hard == 0)
            fpr_hard = fp_hard / (fp_hard + tn_hard) if (fp_hard + tn_hard) > 0 else 0
        else:
            fpr_hard = 0
            
        results.append({
            'Model': model_name.capitalize(),
            'Baseline FPR': fpr_base * 100,
            'Hardened FPR': fpr_hard * 100
        })
            
    return pd.DataFrame(results)

def plot_results(df_results):
    os.makedirs('figures', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_results['Model']))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df_results['Baseline FPR'], width, label='Baseline FPR (pre-hardening)', color='#d62728')
    rects2 = ax.bar(x + width/2, df_results['Hardened FPR'], width, label='Hardened FPR (post-hardening)', color='#2ca02c')
    
    ax.set_ylabel('False Positive Rate (%)')
    ax.set_title('Impact of Non-Adversarial Drift on False Positive Rate', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['Model'], rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=13)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    # Optional styling for better contrast
    ax.set_ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig('figures/drift_vulnerability_plot.pdf', format='pdf', bbox_inches='tight', dpi=600)
    print("Saved plot to figures/drift_vulnerability_plot.pdf")

if __name__ == "__main__":
    df_res = evaluate_models()
    print("Evaluation Results:\n", df_res)
    plot_results(df_res)
