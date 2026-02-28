import pandas as pd
import numpy as np
from scipy.stats import t, f_oneway
import os

def load_evaluation_data():
    base_path = "../output/evaluation/results"
    
    details = pd.read_csv(f'{base_path}/robustness_details.csv')
    imperceptibility = pd.read_csv(f'{base_path}/imperceptibility.csv')
    
    print(f"Loaded {len(details)} robustness measurements")
    return details, imperceptibility

def calculate_confidence_intervals(details):
    results = []
    
    for (attack, param), group in details.groupby(['attack_name', 'attack_param']):
        n = len(group)
        accuracies = group['accuracy'].values
        
        # TPR = accuracy, BER = 1 - accuracy
        tpr = np.mean(accuracies)
        tpr_std = np.std(accuracies, ddof=1)
        ber = 1 - tpr
        
        # 95% confidence intervals using t-distribution
        alpha = 0.05
        df = n - 1
        t_critical = t.ppf(1 - alpha/2, df)
        
        tpr_margin = t_critical * (tpr_std / np.sqrt(n))
        ber_margin = tpr_margin  # Same margin since BER = 1 - TPR
        
        # Ensure bounds are valid [0, 1]
        tpr_ci_lower = max(0, tpr - tpr_margin)
        tpr_ci_upper = min(1, tpr + tpr_margin)
        ber_ci_lower = max(0, ber - ber_margin)
        ber_ci_upper = min(1, ber + ber_margin)
        
        results.append({
            'attack_name': attack,
            'attack_param': param,
            'tpr': tpr,
            'tpr_ci_lower': tpr_ci_lower,
            'tpr_ci_upper': tpr_ci_upper,
            'ber_mean': ber,
            'ber_ci_lower': ber_ci_lower,
            'ber_ci_upper': ber_ci_upper,
            'n_samples': n,
            't_critical': t_critical,
            'margin_tpr': tpr_margin,
            'margin_ber': ber_margin
        })
    
    return pd.DataFrame(results)

def anova_analysis(details):
    attack_groups = {}
    for attack_type in details['attack_name'].unique():
        mask = details['attack_name'] == attack_type
        attack_groups[attack_type] = details[mask]['accuracy'].values
    
    anova_groups = [group for group in attack_groups.values()]
    f_stat, p_value = f_oneway(*anova_groups)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_between': len(attack_groups) - 1,
        'df_within': len(details) - len(attack_groups)
    }

def save_results(ci_df, anova_results):
    output_dir = '../output/evaluation/results'
    
    ci_df.to_csv(f'{output_dir}/confidence_intervals.csv', index=False)
    tpr_df = ci_df[['attack_name', 'attack_param', 'tpr', 'tpr_ci_lower', 'tpr_ci_upper', 'n_samples']].copy()
    tpr_df.to_csv(f'{output_dir}/tpr_fpr_analysis.csv', index=False)
    
    print(f"Results saved to {output_dir}/")
    return True

def print_summary(ci_df, anova_results, imperceptibility):
    print(f"Total configurations analyzed: {len(ci_df)}")
    print(f"Sample size per configuration: {ci_df['n_samples'].iloc[0]}")
    
    print(f"\nTPR Range:")
    print(f"Minimum: {ci_df['tpr'].min():.4f} ({ci_df.loc[ci_df['tpr'].idxmin(), 'attack_name']} attack)")
    print(f"Maximum: {ci_df['tpr'].max():.4f} ({ci_df.loc[ci_df['tpr'].idxmax(), 'attack_name']} attack)")
    
    print(f"\nANOVA Results:")
    print(f"F({anova_results['df_between']},{anova_results['df_within']}) = {anova_results['f_statistic']:.3f}")
    print(f"p-value = {anova_results['p_value']:.2e}")
    
    print(f"\nImperceptibility Metrics:")
    fid = imperceptibility['fid_score'].iloc[0]
    clip_diff = abs(imperceptibility['clip_score_ground_truth'].iloc[0] - imperceptibility['clip_score_watermarked'].iloc[0])
    print(f"FID Score: {fid:.2f}")
    print(f"CLIP Score Difference: {clip_diff:.4f}")

def main():
    details, imperceptibility = load_evaluation_data()
    ci_df = calculate_confidence_intervals(details)
    anova_results = anova_analysis(details)
    save_results(ci_df, anova_results)
    print_summary(ci_df, anova_results, imperceptibility)
    
    print("\nResults saved to CSV files.")

if __name__ == "__main__":
    main()