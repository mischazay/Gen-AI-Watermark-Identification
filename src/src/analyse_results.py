import argparse
import os
import sys
import yaml
import pandas as pd

def analyse_robustness_results(config):
    """
    Analyses the detailed robustness results to create a summary table.
    """
    print("Analysing robustness results...")
    paths = config['paths']
    results_dir = paths['results_dir']
    details_csv_path = os.path.join(results_dir, 'robustness_details.csv')
    summary_csv_path = os.path.join(results_dir, 'robustness_summary.csv')

    # Load the detailed results
    try:
        df = pd.read_csv(details_csv_path)
        print(f"Loaded {len(df)} records from {details_csv_path}")
    except Exception as e:
        print(f"Error loading detailed robustness results: {e}")
        sys.exit(1)

    # Group by attack and calculate mean metrics
    df['attack_param'] = pd.to_numeric(df['attack_param'], errors='coerce')
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['ber'] = pd.to_numeric(df['ber'], errors='coerce')

    # Group and aggregate
    summary_df = df.groupby(['attack_name', 'attack_param']).agg(
        mean_accuracy=('accuracy', 'mean'),
        mean_ber=('ber', 'mean'),
        image_count=('image_file', 'count')
    ).reset_index()

    # Save the summary to a new CSV
    summary_df.sort_values(by=['attack_name', 'attack_param'], inplace=True)
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')

    print(f"\nRobustness summary saved to {summary_csv_path}")
    print("Analysis complete.")
    print("\nRobustness Summary:")
    print(summary_df.to_string())


def main():
    """
    Main function to run the analysis script.
    """
    parser = argparse.ArgumentParser(description="Analyse results from the robustness evaluation.")
    parser.add_argument('config_path', type=str, help='Path to the master configuration file.')
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    analyse_robustness_results(config)

if __name__ == '__main__':
    main()