import json
import random
import unicodedata
import pandas as pd
import codecs
from pathlib import Path
from transformers import CLIPTokenizer
from tqdm import tqdm

def get_tokenizer(model_path="stabilityai/stable-diffusion-2-1-base"):
    """Initialises and returns the CLIP tokenizer."""
    try:
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def is_prompt_valid(prompt, tokenizer, max_length=75):
    """Checks if a prompt is within the tokenizer's token limit."""
    if not isinstance(prompt, str):
        return False
    tokens = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    return len(tokens.input_ids[0]) <= max_length

def clean_prompt(text):
    """
    Clean and normalise prompt text.
    Handles unicode escapes, html entities, and other common issues.
    """
    if not isinstance(text, str):
        return ""

    try:
        text = codecs.decode(text, 'unicode_escape')
    except Exception:
        text = text.replace('\\"', '"')

    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    return text

def curate_coco_prompts(raw_path, output_path, tokenizer):
    """
    Loads all COCO validation annotations, filters them by the 77-token limit,
    assigns random seeds, and saves them to a master JSON file.
    """
    print("Curating all valid COCO prompts...")
    if not tokenizer:
        print("Tokenizer not available. Aborting COCO curation.")
        return

    try:
        with open(raw_path, 'r') as f:
            data = json.load(f)
        
        captions = [item['caption'] for item in data['annotations']]
        unique_captions_raw = list(dict.fromkeys(captions))
        cleaned_captions = [clean_prompt(caption) for caption in unique_captions_raw]
        unique_captions = list(dict.fromkeys(cleaned_captions))
        
        print(f"Found {len(unique_captions)} unique captions. Filtering by token length...")
        
        valid_prompts = [
            caption for caption in tqdm(unique_captions) if is_prompt_valid(caption, tokenizer)
        ]
            
        output_data = [{"prompt": prompt, "seed": random.randint(0, 2**32 - 1)} for prompt in valid_prompts]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully created {output_path} with {len(output_data)} valid prompts.")
        
    except Exception as e:
        print(f"Error during COCO prompt curation: {e}")

def curate_gustavosta_prompts(train_path, test_path, output_path, tokenizer):
    """
    Loads Gustavosta train and test prompts, combines them, filters by the
    77-token limit, assigns random seeds, and saves them to a master JSON file.
    """
    print("Curating all valid Gustavosta prompts...")
    if not tokenizer:
        print("Tokenizer not available. Aborting Gustavosta curation.")
        return
        
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        all_prompts = pd.concat([df_train['Prompt'], df_test['Prompt']], ignore_index=True)
        unique_raw_prompts = all_prompts.dropna().unique().tolist()
        cleaned_prompts = [clean_prompt(prompt) for prompt in unique_raw_prompts]
        unique_prompts = list(dict.fromkeys(cleaned_prompts))

        print(f"Found {len(unique_prompts)} unique prompts from train and test sets. Filtering by token length...")

        valid_prompts = [
            prompt for prompt in tqdm(unique_prompts) if is_prompt_valid(prompt, tokenizer)
        ]

        output_data = [
            {"prompt": prompt, "seed": random.randint(0, 2**32 - 1)}
            for prompt in valid_prompts
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully created {output_path} with {len(output_data)} valid prompts.")

    except Exception as e:
        print(f"Error during Gustavosta prompt curation: {e}")


def main():
    """Main function to run prompt curation for both datasets."""
    base_dir = Path(__file__).resolve().parent.parent
    tokenizer = get_tokenizer()
    
    if not tokenizer:
        print("Failed to initialise tokenizer. Exiting.")
        return

    # Define all paths
    paths = {
        'coco_raw': base_dir / 'data' / 'raw' / 'coco_annotations_trainval2017' / 'captions_val2017.json',
        'coco_output': base_dir / 'data' / 'prompts' / 'coco_prompts_master.json',
        'gustavosta_train': base_dir / 'data' / 'raw' / 'gustavosta' / 'train.csv',
        'gustavosta_test': base_dir / 'data' / 'raw' / 'gustavosta' / 'test.csv',
        'gustavosta_output': base_dir / 'data' / 'prompts' / 'gustavosta_prompts_master.json'
    }
    
    # run for both sets
    curate_coco_prompts(paths['coco_raw'], paths['coco_output'], tokenizer)
    curate_gustavosta_prompts(paths['gustavosta_train'], paths['gustavosta_test'], paths['gustavosta_output'], tokenizer)


if __name__ == "__main__":
    main()