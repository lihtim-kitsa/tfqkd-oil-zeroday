import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar

def bootstrap_metrics(y_true, y_pred, n_bootstraps=2000, random_seed=42):
    """
    Compute 95% Confidence Intervals for Accuracy, F1-macro, and FPR using Bootstrap resampling.
    """
    np.random.seed(random_seed)
    n = len(y_true)
    bootstrapped_acc = []
    bootstrapped_f1 = []
    bootstrapped_fpr = []
    
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n, n)
        y_true_b = y_true[indices]
        y_pred_b = y_pred[indices]
        
        bootstrapped_acc.append(accuracy_score(y_true_b, y_pred_b))
        bootstrapped_f1.append(f1_score(y_true_b, y_pred_b, average='macro', zero_division=0))
        
        # Calculate FPR (False Positive Rate) - assuming binary normal vs attack (label > 0 is attack)
        y_true_binary = (y_true_b > 0).astype(int)
        y_pred_binary = (y_pred_b > 0).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fpr = 0
        bootstrapped_fpr.append(fpr)
        
    def get_ci(metric_list):
        lower = np.percentile(metric_list, 2.5)
        upper = np.percentile(metric_list, 97.5)
        mean_val = np.mean(metric_list)
        return mean_val, lower, upper
        
    return {
        'accuracy': get_ci(bootstrapped_acc),
        'f1_macro': get_ci(bootstrapped_f1),
        'fpr': get_ci(bootstrapped_fpr)
    }

def run_mcnemar_tests(models_preds, y_true):
    """
    Run pairwise McNemar tests across all models and apply Holm-Bonferroni correction.
    models_preds: dict mapping model_name -> y_pred array
    """
    model_names = list(models_preds.keys())
    p_values = []
    comparisons = []
    effect_sizes = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            pred1 = models_preds[name1]
            pred2 = models_preds[name2]
            
            # Contingency table for McNemar
            # Cell (0,0): both correct
            # Cell (0,1): 1 correct, 2 wrong
            # Cell (1,0): 1 wrong, 2 correct
            # Cell (1,1): both wrong
            correct1 = (pred1 == y_true)
            correct2 = (pred2 == y_true)
            
            n00 = np.sum(correct1 & correct2)
            n01 = np.sum(correct1 & ~correct2)
            n10 = np.sum(~correct1 & correct2)
            n11 = np.sum(~correct1 & ~correct2)
            
            table = [[n00, n01], [n10, n11]]
            result = mcnemar(table, exact=False, correction=True)
            p_values.append(result.pvalue)
            comparisons.append(f"{name1} vs {name2}")
            
            # Effect size (odds ratio = n01 / n10)
            if n10 == 0:
                odds_ratio = float('inf')
            else:
                odds_ratio = n01 / n10
            effect_sizes.append(odds_ratio)
            
    # Holm-Bonferroni correction
    if len(p_values) > 0:
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
    else:
        reject = []
        pvals_corrected = []
        
    results = {}
    for comp, p_raw, p_adj, rej, eff in zip(comparisons, p_values, pvals_corrected, reject, effect_sizes):
        results[comp] = {
            'p_raw': p_raw,
            'p_holm_corrected': p_adj,
            'significant': rej,
            'odds_ratio_effect_size': eff
        }
        
    return results
