import numpy as np
from statsmodels.stats.power import TTestIndPower

def calculate_power():
    """
    Calculate power for testing the difference between two models' accuracies.
    Assuming we want to detect a 2 percentage point difference (e.g. 90% vs 92%).
    """
    # Effect size h for proportions (Cohen's h)
    p1 = 0.90
    p2 = 0.92
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    
    alpha = 0.05
    power = 0.80
    
    analysis = TTestIndPower()
    
    # calculate sample size
    sample_size = analysis.solve_power(effect_size=abs(h), power=power, alpha=alpha)
    
    print("--- Statistical Power Analysis ---")
    print(f"Baseline accuracy: {p1:.1%}")
    print(f"Target accuracy:   {p2:.1%}")
    print(f"Alpha level:       {alpha}")
    print(f"Target power:      {power}")
    print(f"Cohen's h:         {abs(h):.4f}")
    print(f"Required samples per group (test set): {np.ceil(sample_size):.0f}")
    print(f"Total test set needed (assuming 2 groups): {np.ceil(sample_size) * 2:.0f}")
    
    print("\nConclusion: With a 40,000 sample dataset and an 80/20 train/test split (8,000 test samples),")
    print("we have more than enough statistical power to detect a 2 percentage point difference.")

if __name__ == "__main__":
    calculate_power()
