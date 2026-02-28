import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys
from PIL import Image
import yaml

def set_plot_style():
    """Sets a professional, publication-quality style for the plots."""
    plt.style.use('seaborn-v0_8-whitegrid')

def generate_visual_comparison_figure(config, image_indices, output_path):
    """
    Generates a 2xN grid of images for visual comparison.
    
    Args:
        config (dict): The project configuration.
        image_indices (list): A list of integer indices for the images to include.
        output_path (str): The path to save the final figure.
    """
    print(f"Generating visual comparison for indices: {image_indices}...")
    
    paths = config['paths']
    gt_dir = paths['ground_truth_coco_dir']
    wm_dir = paths['watermarked_coco_dir']
    
    num_images = len(image_indices)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 4, 8))
    
    # Set titles for rows
    axes[0, 0].set_ylabel('Non-Watermarked', fontsize=12, labelpad=10)
    axes[1, 0].set_ylabel('Watermarked', fontsize=12, labelpad=10)
    
    for i, img_index in enumerate(image_indices):
        gt_filename = f"{img_index:04d}.png"
        wm_filename = f"{img_index:04d}.png"
        
        gt_path = os.path.join(gt_dir, gt_filename)
        wm_path = os.path.join(wm_dir, wm_filename)
        
        try:
            # original Image
            gt_img = Image.open(gt_path)
            axes[0, i].imshow(gt_img)
            axes[0, i].axis('off')
            
            # Load Watermarked Image
            wm_img = Image.open(wm_path)
            axes[1, i].imshow(wm_img)
            axes[1, i].axis('off')
            
        except FileNotFoundError as e:
            print(f"Error: Could not find image file for index {img_index}. {e}")
            # Create a placeholder for missing images
            axes[0, i].text(0.5, 0.5, f'Image {img_index}\nNot Found', ha='center', va='center')
            axes[1, i].text(0.5, 0.5, f'Image {img_index}\nNot Found', ha='center', va='center')
            axes[0, i].axis('off')
            axes[1, i].axis('off')

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Visual comparison figure saved to {output_path}")

def generate_robustness_plots(results_path, output_dir):
    """
    Generates and saves the main robustness plots for the report.
    """
    print("Generating robustness plots...")
    
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Fig 1
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1 JPEG Compression
    jpeg_df = df[df['attack_name'] == 'jpeg'].sort_values('attack_param', ascending=False)
    ax1.plot(jpeg_df['attack_param'], jpeg_df['mean_ber'], marker='o', linestyle='-')
    ax1.set_title('Robustness to JPEG Compression')
    ax1.set_xlabel('JPEG Quality Factor')
    ax1.set_ylabel('Mean Bit Error Rate (BER)')
    ax1.set_ylim(0, max(jpeg_df['mean_ber'].max() * 1.2, 0.02)) 
    ax1.invert_xaxis() 

    # gaussian Noise
    noise_df = df[df['attack_name'] == 'gaussian_noise'].sort_values('attack_param')
    ax2.plot(noise_df['attack_param'], noise_df['mean_ber'], marker='o', linestyle='-')
    ax2.set_title('Robustness to Gaussian Noise')
    ax2.set_xlabel('Noise Standard Deviation ($\sigma$)')
    ax2.set_ylabel('Mean Bit Error Rate (BER)')
    ax2.set_ylim(0, max(noise_df['mean_ber'].max() * 1.2, 0.02))

    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, 'robustness_jpeg_noise.pdf')
    fig1.savefig(fig1_path, bbox_inches='tight')
    print(f"Saved JPEG & Noise plot to {os.path.abspath(fig1_path)}")

    # Fig 2
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))

    # gaussian Blur
    blur_df = df[df['attack_name'] == 'gaussian_blur'].sort_values('attack_param')
    ax3.plot(blur_df['attack_param'], blur_df['mean_ber'], marker='o', linestyle='-')
    ax3.set_title('Robustness to Gaussian Blur')
    ax3.set_xlabel('Blur Radius')
    ax3.set_ylabel('Mean Bit Error Rate (BER)')
    ax3.set_ylim(0, max(blur_df['mean_ber'].max() * 1.2, 0.03))

    # Comparison of Attacks
    crop_ber = df[df['attack_name'] == 'crop']['mean_ber'].iloc[0]
    jpeg_25_ber = df[(df['attack_name'] == 'jpeg') & (df['attack_param'] == 25)]['mean_ber'].iloc[0]
    noise_05_ber = df[(df['attack_name'] == 'gaussian_noise') & (df['attack_param'] == 0.05)]['mean_ber'].iloc[0]
    blur_4_ber = df[(df['attack_name'] == 'gaussian_blur') & (df['attack_param'] == 4)]['mean_ber'].iloc[0]

    attack_comp_data = {
        'Attack': ['JPEG (QF=25)', 'Noise ($\sigma=0.05$)', 'Blur (R=4)', 'Crop (80%)'],
        'BER': [jpeg_25_ber, noise_05_ber, blur_4_ber, crop_ber]
    }
    attack_comp_df = pd.DataFrame(attack_comp_data)
    
    sns.barplot(x='Attack', y='BER', data=attack_comp_df, ax=ax4, palette='viridis', hue='Attack', legend=False)
    ax4.set_title('Robustness to Severe & Geometric Attacks')
    ax4.set_ylabel('Mean Bit Error Rate (BER)')
    ax4.set_xlabel('Attack Type')
    ax4.set_xticks(range(len(attack_comp_df)))
    ax4.set_xticklabels(attack_comp_df['Attack'], rotation=15, ha='right')
    ax4.patches[-1].set_facecolor('salmon')
    ax4.patches[-1].set_edgecolor('red')


    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, 'robustness_blur_crop.pdf')
    fig2.savefig(fig2_path, bbox_inches='tight')
    print(f"Saved Blur & Crop plot to {os.path.abspath(fig2_path)}")

def main():
    """Main function to generate all report figures."""
    parser = argparse.ArgumentParser(description='Generate all figures for the report.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output figures.')
    parser.add_argument('--results_csv', type=str, default='Code/Gaussian-Shading/output/evaluation/results/robustness_summary.csv', help='Path to the robustness summary CSV file.')
    parser.add_argument('--config', type=str, default='Code/Gaussian-Shading/configs/master_config.yaml', help='Path to the master config YAML file.')
    parser.add_argument('--image_indices', type=int, nargs='+', default=[10, 25, 50], help='List of image indices to use for visual comparison.')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating the robustness plots.')
    parser.add_argument('--skip-visuals', action='store_true', help='Skip generating the visual comparison figure.')
    args = parser.parse_args()

    set_plot_style()
    
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    generated_figures = []

    if not args.skip_plots:
        results_path = os.path.abspath(args.results_csv)
        generate_robustness_plots(results_path, output_dir)
        generated_figures.append("robustness plots")

    if not args.skip_visuals:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: {args.config}")
            sys.exit(1)
        
        visual_output_path = os.path.join(output_dir, 'visual_comparison.pdf')
        generate_visual_comparison_figure(config, args.image_indices, visual_output_path)
        generated_figures.append("visual comparison")

    if generated_figures:
        print(f"\nSuccessfully generated: {', '.join(generated_figures)}.")
    else:
        print("\nNo figures generated")

if __name__ == '__main__':
    main()
