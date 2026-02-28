import argparse
import os
import sys
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.attack_utils import get_attack_func

def main():
    """
    Main function to apply a suite of attacks to a directory of watermarked images.
    """
    parser = argparse.ArgumentParser(description="Apply attacks to watermarked images.")
    parser.add_argument('config_path', type=str, help='Path to the master configuration file.')
    parser.add_argument('--dataset', type=str, required=True, choices=['gustavosta'], help='The dataset to process.')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of images to process for testing.')
    args = parser.parse_args()

    # Load Configuration
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Setup Paths
    dataset = args.dataset
    try:
        source_dir = config['paths'][f'watermarked_{dataset}_dir']
        base_attack_dir = config['paths']['attacked_dir']
    except KeyError as e:
        print(f"Error: Missing path configuration for dataset '{dataset}'. Missing key: {e}")
        sys.exit(1)
        
    print(f"Source directory: {source_dir}")
    print(f"Base attack directory: {base_attack_dir}")

    # Load Image Paths
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        sys.exit(1)
    
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {source_dir}. Exiting.")
        return

    # Apply limit if set in args
    if args.limit is not None:
        print(f"Limiting images to first {args.limit}")
        image_files = image_files[:args.limit]

    # Apply Attacks
    attacks_to_run = config.get('attacks', {})
    if not attacks_to_run:
        print("No attacks defined in the configuration file. Exiting.")
        return

    for attack_name, params in attacks_to_run.items():
        attack_func = get_attack_func(attack_name)
        if not attack_func:
            print(f"Warning: No attack function found for '{attack_name}'. Skipping.")
            continue

        for param in params:
            # Create a descriptive subdirectory for each attack and parameter
            attack_subdir_name = f"{attack_name}_{str(param).replace('.', 'p')}"
            output_dir = os.path.join(base_attack_dir, dataset, attack_subdir_name)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\nApplying attack: {attack_name} with parameter: {param}")
            print(f"Saving results to: {output_dir}")

            for image_file in tqdm(image_files, desc=f"Processing {attack_subdir_name}"):
                source_path = os.path.join(source_dir, image_file)
                dest_path = os.path.join(output_dir, image_file)

                # skips if attacked image already exists
                if os.path.exists(dest_path):
                    continue

                try:
                    with Image.open(source_path) as img:
                        img_rgb = img.convert("RGB")
                        attacked_image = attack_func(img_rgb, param)
                        attacked_image.save(dest_path)
                except Exception as e:
                    print(f"Error processing {image_file} for attack {attack_name}: {e}")

    print("\nAll attacks have been applied.")

if __name__ == '__main__':
    main()