import argparse
import json
import os
import random
import sys
import torch
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='diffusers.modeling_utils') # ignore diffuser warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.modeling_utils')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from image_utils import set_random_seed
from watermark import Gaussian_Shading

def generate_coco_ground_truth(config, num_images=1000):
    """
    Generate ground truth images for COCO dataset if they don't already exist.
    Returns the metadata for the generated images.
    """
    
    prompts_path = config['paths']['coco_prompts_file']
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)

    output_dir = config['paths']['ground_truth_coco_dir']
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []

    num_already_generated = len(metadata)
    target_num_images = num_images
    num_to_generate = target_num_images - num_already_generated

    if num_to_generate <= 0:
        print(f"Target of {target_num_images} COCO ground truth images already met ({num_already_generated} found).")
        return metadata

    print(f"Generating {num_to_generate} COCO ground truth images...")

    existing_prompts = {item['prompt'] for item in metadata}
    available_prompts = [p for p in prompts_data if p['prompt'] not in existing_prompts]
    
    if len(available_prompts) < num_to_generate:
        print(f"Warning: Only {len(available_prompts)} new unique prompts available, but {num_to_generate} are needed.")
        print(f"Generating {len(available_prompts)} images instead.")
        num_to_generate = len(available_prompts)

    # Reproducible random sampling of the required number of new prompts
    random.seed(42)
    sampled_prompts = random.sample(available_prompts, num_to_generate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_config = config['model']
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_config['name'], subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_config['name'],
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    start_index = len(metadata)
    
    generated_files = {f for f in os.listdir(output_dir) if f.endswith('.png')}

    prompts_to_generate = []
    for i, prompt_info in enumerate(sampled_prompts):
        image_index = start_index + i
        image_filename = f"{image_index:04d}.png"
        if image_filename not in generated_files:
            prompts_to_generate.append((image_index, prompt_info))

    if not prompts_to_generate:
        print("All required COCO ground truth images have already been generated.")
        return metadata

    print(f"Generating {len(prompts_to_generate)} new COCO ground truth images...")

    for image_index, prompt_info in tqdm(prompts_to_generate, desc="Generating COCO Ground Truth"):
        prompt = prompt_info['prompt']
        seed = prompt_info['seed']
        
        set_random_seed(seed)

        with torch.no_grad():
            image = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=config['generation']['guidance_scale'],
                num_inference_steps=config['generation']['inference_steps'],
                height=model_config['resolution'],
                width=model_config['resolution'],
            ).images[0]

        image_path = os.path.join(output_dir, f"{image_index:04d}.png")
        image.save(image_path)
        metadata.append({
            "image_path": os.path.basename(image_path),
            "prompt": prompt,
            "seed": seed
        })
        
        # Save metadata incrementally
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    print(f"Successfully generated {len(prompts_to_generate)} COCO ground truth images.")
    return metadata

def main():
    """
    Main function to run the image generation process.
    """
    parser = argparse.ArgumentParser(description="Generate watermarked and ground truth images for the main experiment.")
    parser.add_argument('config_path', type=str, help='Path to the master configuration file.')
    parser.add_argument('--dataset', type=str, required=True, choices=['coco', 'gustavosta'], help='The dataset to use for generation (coco or gustavosta).')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of prompts to process for testing.')
    parser.add_argument('--coco_images', type=int, default=1000, help='Number of COCO ground truth images to generate (default: 1000).')
    args = parser.parse_args()

    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    dataset = args.dataset
    print(f"Running for dataset: {dataset}")

    try:
        prompts_path = config['paths'][f'{dataset}_prompts_file']
        watermarked_dir = config['paths'][f'watermarked_{dataset}_dir']
        ground_truth_dir = config['paths'][f'ground_truth_{dataset}_dir']
    except KeyError as e:
        print(f"Error: Missing path configuration for dataset '{dataset}'. Missing key: {e}")
        sys.exit(1)

    os.makedirs(watermarked_dir, exist_ok=True)
    if dataset == 'gustavosta':
        os.makedirs(ground_truth_dir, exist_ok=True)

    if dataset == 'coco':
        print("Processing COCO dataset...")
        coco_metadata = generate_coco_ground_truth(config, num_images=args.coco_images)
        
        if args.limit is not None:
            print(f"Limiting COCO prompts to first {args.limit}")
            coco_metadata = coco_metadata[:args.limit]
        
        prompts_data = coco_metadata
        print(f"Using {len(prompts_data)} COCO prompts for watermarked generation")
        
    else:
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
        except Exception as e:
            print(f"Error loading prompts: {e}")
            sys.exit(1)
        
        # Apply limit from args
        if args.limit is not None:
            print(f"Limiting Gustavosta prompts to first {args.limit}")
            prompts_data = prompts_data[:args.limit]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_config = config['model']
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_config['name'], subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_config['name'],
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    wm_config = config['watermark']
    watermarker = Gaussian_Shading(
        ch_factor=wm_config['channel_copy'],
        hw_factor=wm_config['hw_copy'],
        fpr=wm_config['detection_fpr'],
        user_number=1_000_000
    )

    # generation Loop
    gen_config = config['generation']
    metadata_records = []

    if args.dataset == 'gustavosta':
        gt_metadata_path = os.path.join(config['paths'][f'ground_truth_{args.dataset}_dir'], 'metadata.json')
        if os.path.exists(gt_metadata_path):
            try:
                with open(gt_metadata_path, 'r', encoding='utf-8') as f:
                    metadata_records = json.load(f)
                print(f"Loaded {len(metadata_records)} existing metadata records for gustavosta.")
            except Exception as e:
                print(f"Could not read existing metadata: {e}. Starting from scratch.")
                metadata_records = []

    for i, prompt_info in enumerate(tqdm(prompts_data, desc="Processing Prompts")):
        if dataset == 'coco':
            prompt_text = prompt_info['prompt']
            seed = prompt_info['seed']
            base_filename = prompt_info['image_path'] 
        else: # gustavosta
            prompt_text = prompt_info['prompt']
            seed = prompt_info['seed']
            base_filename = f"{i:04d}.png"

        watermarked_path = os.path.join(watermarked_dir, base_filename)
        ground_truth_path = os.path.join(ground_truth_dir, base_filename)

        if dataset == 'gustavosta' and any(rec['image_path'] == base_filename for rec in metadata_records):
            continue
        
        # Check if files already exist
        if dataset == 'coco':
            if os.path.exists(watermarked_path):
                print(f"Skipping generation for {base_filename} (seed: {seed}): Watermarked image already exists.")
                continue
        else:
            if os.path.exists(watermarked_path) and os.path.exists(ground_truth_path):
                print(f"Skipping generation for {base_filename} (seed: {seed}): Files already exist.")
                continue

        print(f"Generating images for {base_filename} (seed: {seed})...")
        
        if dataset == 'gustavosta':
            set_random_seed(seed)
            device = pipe.device
            with torch.no_grad():
                image_gt = pipe(
                    prompt_text,
                    num_images_per_prompt=1,
                    guidance_scale=gen_config['guidance_scale'],
                    num_inference_steps=gen_config['inference_steps'],
                    height=model_config['resolution'],
                    width=model_config['resolution'],
                ).images[0]
            image_gt.save(ground_truth_path)
            print(f"  - Saved ground truth image to {ground_truth_path}")

            set_random_seed(seed)
            init_latents_w = watermarker.create_watermark_and_return_w()
            with torch.no_grad():
                image_w = pipe(
                    prompt_text,
                    num_images_per_prompt=1,
                    guidance_scale=gen_config['guidance_scale'],
                    num_inference_steps=gen_config['inference_steps'],
                    height=model_config['resolution'],
                    width=model_config['resolution'],
                    latents=init_latents_w,
                ).images[0]
            image_w.save(watermarked_path)
            print(f"  - Saved watermarked image to {watermarked_path}")
            
        else:
            print(f"  - Ground truth image already exists at {ground_truth_path}")
            
            set_random_seed(seed)
            init_latents_w = watermarker.create_watermark_and_return_w()
            with torch.no_grad():
                image_w = pipe(
                    prompt_text,
                    num_images_per_prompt=1,
                    guidance_scale=gen_config['guidance_scale'],
                    num_inference_steps=gen_config['inference_steps'],
                    height=model_config['resolution'],
                    width=model_config['resolution'],
                    latents=init_latents_w,
                ).images[0]
            image_w.save(watermarked_path)
            print(f"  - Saved watermarked image to {watermarked_path}")

        # Save metadata for generated images
        if dataset == 'gustavosta':
            if not any(d['image_path'] == base_filename for d in metadata_records):
                metadata_records.append({'prompt': prompt_text, 'seed': seed, 'image_path': base_filename})

            # Save whole metadata list on each iteration
            gt_metadata_path = os.path.join(ground_truth_dir, 'metadata.json')
            wm_metadata_path = os.path.join(watermarked_dir, 'metadata.json')
            
            with open(gt_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_records, f, indent=4)
            
            with open(wm_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_records, f, indent=4)
                
        else:
            wm_metadata_path = os.path.join(watermarked_dir, 'metadata.json')
            if os.path.exists(wm_metadata_path):
                with open(wm_metadata_path, 'r', encoding='utf-8') as f:
                    wm_metadata = json.load(f)
            else:
                wm_metadata = []
            if not any(d['image_path'] == base_filename for d in wm_metadata):
                wm_metadata.append({'prompt': prompt_text, 'seed': seed, 'image_path': base_filename})
                with open(wm_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(wm_metadata, f, indent=4)

    if dataset == 'coco':
        print(f"\nCOCO processing complete")
    else:
        print(f"\nGustavosta dataset complete")

if __name__ == '__main__':
    main()