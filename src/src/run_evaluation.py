import argparse
import json
import os
import sys
import yaml
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import open_clip
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='diffusers.modeling_utils')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.modeling_utils')

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermark import Gaussian_Shading
from image_utils import transform_img

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def calculate_ber(original_bits, extracted_bits):
    """Calculates the Bit Error Rate (BER) between two bitstrings."""
    if len(original_bits) != len(extracted_bits):
        raise ValueError("Bitstrings must have the same length.")
    return np.sum(original_bits != extracted_bits) / len(original_bits)

def optimise_gpu_memory():
    """Optimise GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass

def batch_process_images(image_data_list, pipe, watermarker, text_embeddings, device, batch_size=12):
    """
    Process images in batches to maximise GPU utilisation.
    Each image_data item should be (image_path, seed).
    Returns list of (accuracy, ber) tuples.
    """
    results = []
    
    for i in range(0, len(image_data_list), batch_size):
        batch_data = image_data_list[i:i + batch_size]
        batch_images = []
        batch_tensors = []
        batch_seeds = []
        
        for img_path, seed in batch_data:
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
                image_tensor = transform_img(image).unsqueeze(0).to(text_embeddings.dtype).to(device)
                batch_tensors.append(image_tensor)
                batch_seeds.append(seed)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                results.append((0.0, 1.0))  # Failed processing
                continue
        
        if not batch_tensors:
            continue
            
        # Batch process tensors with memory optimisation
        try:
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                # Stack tensors for batch processing
                if len(batch_tensors) > 1:
                    batch_tensor = torch.cat(batch_tensors, dim=0)
                else:
                    batch_tensor = batch_tensors[0]
                
                # Batch encode to latents
                image_latents = pipe.get_image_latents(batch_tensor, sample=False)
                
                # DDIM inversion
                reversed_latents = pipe.forward_diffusion(
                    latents=image_latents, 
                    text_embeddings=text_embeddings.expand(batch_tensor.shape[0], -1, -1),
                    guidance_scale=1, 
                    num_inference_steps=50
                )
                
                # Process each item in the batch with its correct watermark
                for j in range(reversed_latents.shape[0]):
                    seed = batch_seeds[j]
                    latent_slice = reversed_latents[j:j+1]
                    
                    # Set up the correct watermark for this specific image
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    _ = watermarker.create_watermark_and_return_w()
                    
                    # Now evaluate with the correct watermark
                    accuracy = watermarker.eval_watermark(latent_slice)
                    ber = 1.0 - accuracy
                    results.append((accuracy, ber))
                    
        except Exception as e:
            print(f"Error processing batch: {e}")
            # failed results for this batch
            for _ in batch_data:
                results.append((0.0, 1.0))
    
    return results

def run_imperceptibility_evaluation(config):
    """
    Calculates FID and CLIP scores between ground truth and watermarked images.
    This evaluation compares the distributions of watermarked vs non-watermarked images.
    """
    print("Running Imperceptibility Evaluation...")
    print("Note: Comparing distributions of watermarked vs non-watermarked images")
    paths = config['paths']
    gt_dir = paths['ground_truth_coco_dir']
    wm_dir = paths['watermarked_coco_dir']
    results_dir = paths['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    output_csv = os.path.join(results_dir, 'imperceptibility.csv')

    print(f"Ground Truth Dir: {gt_dir}")
    print(f"Watermarked Dir: {wm_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    optimise_gpu_memory()

    # FID Score calc
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        print("Calculating FID score...")
        print("This measures how similar the distributions of watermarked and non-watermarked images are.")
        fid_score = calculate_fid_given_paths([gt_dir, wm_dir], batch_size=64, device=device, dims=2048)
        print(f"FID Score: {fid_score}")
        
    except Exception as e:
        print(f"Error calculating FID score: {e}")
        fid_score = -1.0

    # CLIP Score
    print("Calculating CLIP scores")
    clip_score_ground_truth = -1.0
    clip_score_watermarked = -1.0
    try:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # Load metadata to get prompts
        metadata_path = os.path.join(gt_dir, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        prompt_lookup = {item['image_path']: item['prompt'] for item in metadata}

        def calculate_avg_clip_score_batched(image_dir, batch_size=12):
            """Optimised CLIP score calculation with batching"""
            total_clip_score = 0.0
            image_count = 0
            
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                batch_images = []
                batch_prompts = []
                
                # Prepare batch
                for filename in batch_files:
                    if filename in prompt_lookup:
                        img_path = os.path.join(image_dir, filename)
                        try:
                            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                            batch_images.append(image)
                            batch_prompts.append(prompt_lookup[filename])
                        except Exception as e:
                            print(f"Could not process {filename} for CLIP score. Error: {e}")
                            continue
                
                if not batch_images:
                    continue
                
                # Batch processing
                try:
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                        # Stack images and encode
                        batch_tensor = torch.cat(batch_images, dim=0)
                        image_features = model.encode_image(batch_tensor)
                        
                        # Tokenise prompts and encode
                        text_tokens = tokenizer(batch_prompts).to(device)
                        text_features = model.encode_text(text_tokens)
                        
                        # Normalise features
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        
                        # Calculate similarities
                        similarities = (image_features * text_features).sum(dim=-1)
                        
                        total_clip_score += similarities.sum().item()
                        image_count += len(batch_images)
                        
                except Exception as e:
                    print(f"Error processing CLIP batch: {e}")
                    continue

            return total_clip_score / image_count if image_count > 0 else 0.0

        print("Calculating CLIP score for ground truth images...")
        clip_score_ground_truth = calculate_avg_clip_score_batched(gt_dir)
        print(f"Average CLIP Score (Ground Truth): {clip_score_ground_truth:.4f}")

        print("Calculating CLIP score for watermarked images...")
        clip_score_watermarked = calculate_avg_clip_score_batched(wm_dir)
        print(f"Average CLIP Score (Watermarked): {clip_score_watermarked:.4f}")

    except Exception as e:
        print(f"Error calculating CLIP scores: {e}")

    # Save Results
    results_df = pd.DataFrame([{
        'fid_score': fid_score,
        'clip_score_ground_truth': clip_score_ground_truth,
        'clip_score_watermarked': clip_score_watermarked
    }])
    results_df.to_csv(output_csv, index=False)
    
    print(f"Imperceptibility results saved to {output_csv}")

def run_robustness_evaluation(config):
    """
    Calculates BER for each attacked image, saving results incrementally to a CSV file.
    This function is designed to be resumable and uses optimised batch processing.
    """
    print("Running Robustness Evaluation...")
    paths = config['paths']
    attacked_base_dir = paths['attacked_dir']
    watermarked_gustavosta_dir = paths['watermarked_gustavosta_dir']
    results_dir = paths['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    output_csv = os.path.join(results_dir, 'robustness_details.csv')
    csv_headers = ['attack_name', 'attack_param', 'image_file', 'accuracy', 'ber']

    # load existing results
    if os.path.exists(output_csv):
        try:
            existing_results_df = pd.read_csv(output_csv)
            print(f"Loaded {len(existing_results_df)} existing results from {output_csv}. Resuming evaluation.")
        except Exception as e:
            existing_results_df = pd.DataFrame(columns=csv_headers)
            print(f"Error loading existing results: {e}. Starting fresh.")
    else:
        existing_results_df = pd.DataFrame(columns=csv_headers)
        existing_results_df.to_csv(output_csv, index=False)
        print(f"Created new results file at {output_csv}.")

    completed_tasks = set(zip(existing_results_df['attack_name'], existing_results_df['attack_param'].astype(float), existing_results_df['image_file']))

    # Load Metadata for Seed Lookup
    metadata_path = os.path.join(watermarked_gustavosta_dir, 'metadata.json')
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        seed_lookup = {item['image_path']: item['seed'] for item in metadata}
    except Exception as e:
        print(f"Error loading metadata: {e}. Cannot run robustness evaluation.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    optimise_gpu_memory()
    
    model_config = config['model']
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_config['name'], subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_config['name'], scheduler=scheduler, torch_dtype=torch.float16, revision='fp16'
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)
    
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    wm_config = config['watermark']
    watermarker = Gaussian_Shading(
        ch_factor=wm_config['channel_copy'], hw_factor=wm_config['hw_copy'],
        fpr=wm_config['detection_fpr'], user_number=1_000_000
    )
    
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # eval Loop
    attacked_gustavosta_dir = os.path.join(attacked_base_dir, 'gustavosta')
    if not os.path.isdir(attacked_gustavosta_dir):
        print(f"Attacked directory for gustavosta not found at {attacked_gustavosta_dir}. Exiting.")
        return

    attack_dirs = sorted([d for d in os.listdir(attacked_gustavosta_dir) if os.path.isdir(os.path.join(attacked_gustavosta_dir, d))])

    for attack_dir_name in tqdm(attack_dirs, desc="Evaluating Attacks"):
        try:
            attack_name, attack_param_str = attack_dir_name.rsplit('_', 1)
            attack_param = attack_param_str.replace('p', '.')
        except ValueError:
            print(f"Warning: Skipping directory with unexpected name format: {attack_dir_name}")
            continue
        
        full_attack_dir = os.path.join(attacked_gustavosta_dir, attack_dir_name)
        image_files = sorted([f for f in os.listdir(full_attack_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        pending_files = []
        for image_file in image_files:
            if (attack_name, float(attack_param), image_file) not in completed_tasks:
                if image_file in seed_lookup:
                    pending_files.append(image_file)
                else:
                    print(f"Warning: Could not find seed for {image_file}. Skipping.")
        
        if not pending_files:
            print(f"All images in {attack_dir_name} already processed. Skipping.")
            continue

        print(f"Processing {len(pending_files)} images in {attack_dir_name}")
        if len(pending_files) > 0:
            print(f"Expected GPU utilisation: {len(pending_files)} images × 50 DDIM steps = {len(pending_files) * 50} GPU operations")

        batch_data = []
        for image_file in pending_files:
            seed = seed_lookup[image_file]
            img_path = os.path.join(full_attack_dir, image_file)
            batch_data.append((image_file, seed, img_path))

        batch_size = 12
        for batch_idx, i in enumerate(range(0, len(batch_data), batch_size)):
            batch = batch_data[i:i + batch_size]
            
            batch_processing_data = []
            image_file_order = []
            for image_file, seed, img_path in batch:
                batch_processing_data.append((img_path, seed))
                image_file_order.append(image_file)
            
            try:
                # process batch
                results = batch_process_images(batch_processing_data, pipe, watermarker, text_embeddings, device, batch_size=len(batch_processing_data))
                
                # Save results for batch
                for j, image_file in enumerate(image_file_order):
                    if j < len(results):
                        accuracy, ber = results[j]
                        
                        # Append result to CSV
                        new_row = pd.DataFrame([{
                            'attack_name': attack_name,
                            'attack_param': attack_param,
                            'image_file': image_file,
                            'accuracy': accuracy,
                            'ber': ber
                        }])
                        new_row.to_csv(output_csv, mode='a', header=False, index=False)
                        
                        # add to completed tasks set
                        completed_tasks.add((attack_name, float(attack_param), image_file))
                        
            except Exception as e:
                print(f"Error processing batch in {attack_dir_name}: {e}")
                
            # Clear GPU cache periodically
            if batch_idx > 0 and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    print("\nRobustness evaluation finished.")
    print(f"Detailed results saved incrementally to {output_csv}")

def main():
    """
    Main function to orchestrate the evaluation process.
    """
    parser = argparse.ArgumentParser(description="Run evaluation for imperceptibility or robustness.")
    parser.add_argument('config_path', type=str, help='Path to the master configuration file.')
    parser.add_argument('--mode', type=str, required=True, choices=['imperceptibility', 'robustness', 'all'], help='The evaluation mode to run.')
    args = parser.parse_args()

    # Load configs
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    if args.mode == 'imperceptibility':
        run_imperceptibility_evaluation(config)
    elif args.mode == 'robustness':
        run_robustness_evaluation(config)
    elif args.mode == 'all':
        print("Running both imperceptibility and robustness evaluations...")
        run_imperceptibility_evaluation(config)
        run_robustness_evaluation(config)

if __name__ == '__main__':
    main()