import pandas as pd
import numpy as np

def run_data_checks():
    df = pd.read_parquet('dataset_v1.parquet')
    
    print("--- DATA ANOMALY CHECK REPORT ---")
    
    # 1. Check for Missing Values
    print("\n1. Missing Values (NaNs):")
    print(df.isnull().sum())
    
    # 2. Check for infinite values
    print("\n2. Infinite Values:")
    print(np.isinf(df).sum())
    
    # 3. Check for Mislabelled Data (Label 0 behaving like Label 1 or 2)
    # We can do this by looking at feature overlap. 
    # Are there extreme outliers in Label 0 that fall right into the mean of Label 1 or 2?
    
    print("\n3. Feature Distribution Overlap (Mean & Std Dev):")
    stats = df.groupby('label').agg(['mean', 'std', 'min', 'max'])
    for feature in ['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']:
        print(f"\nFeature: {feature}")
        print(stats[feature])
        
    # Check for "Label 0" points that look suspiciously like attacks.
    # Let's say, any Label 0 point that is beyond 3 standard deviations from the Label 0 mean, 
    # AND is within 1 standard deviation of the Label 1 or 2 mean.
    
    print("\n4. Searching for potentially mislabelled 'Eve' instances (Label 0):")
    label_0 = df[df['label'] == 0]
    
    potential_mislabels = 0
    for feature in ['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']:
        mean_0 = stats.loc[0, (feature, 'mean')]
        std_0 = stats.loc[0, (feature, 'std')]
        
        # We find points in label 0 that are very far from label 0's mean
        # Let's just look at extreme outliers in label 0
        outliers = label_0[(label_0[feature] > mean_0 + 4 * std_0) | (label_0[feature] < mean_0 - 4 * std_0)]
        if not outliers.empty:
            print(f"  - Found {len(outliers)} extreme outliers in Label 0 for feature '{feature}'")
            potential_mislabels += len(outliers)
            
    if potential_mislabels == 0:
        print("  - No extreme univariate outliers found in Label 0 that might be hidden attacks.")
        
    # Let's also check a simple isolation forest or just distance to means for multidimensional check
    from scipy.spatial.distance import cdist
    
    means_0 = df[df['label'] == 0][['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].mean().values.reshape(1, -1)
    means_1 = df[df['label'] == 1][['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].mean().values.reshape(1, -1)
    means_2 = df[df['label'] == 2][['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']].mean().values.reshape(1, -1)
    
    dist_to_0 = cdist(label_0[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']], means_0)
    dist_to_1 = cdist(label_0[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']], means_1)
    dist_to_2 = cdist(label_0[['mu', 'sigma2', 'Psb', 'delta_phi', 'qber']], means_2)
    
    # Are there any label 0 points that are closer to the mean of label 1 or label 2 than to label 0?
    closer_to_1 = (dist_to_1 < dist_to_0).sum()
    closer_to_2 = (dist_to_2 < dist_to_0).sum()
    
    print(f"\n5. Multidimensional Distance Check for Label 0:")
    print(f"  - Normal samples closer to FIM attack (Label 1) center than Normal center: {closer_to_1}")
    print(f"  - Normal samples closer to TWIRL attack (Label 2) center than Normal center: {closer_to_2}")

if __name__ == '__main__':
    run_data_checks()
