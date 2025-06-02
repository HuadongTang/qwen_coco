import os
import json
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import re
import cv2
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure 4-bit quantization to reduce memory usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto",
    attn_implementation="flash_attention_2"
)

# Default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", use_fast=True)

# Define input and output paths
folder_path = '/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/images/train2017'
out_dir = '/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_object_coco/'
os.makedirs(out_dir, exist_ok=True)

# Get list of valid image files
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

# Check for already processed images
processed_files = set(os.listdir(out_dir))  # Get list of JSON files in output directory
image_files = [f for f in image_files if os.path.splitext(f)[0] + '.json' not in processed_files]

total_images = len(image_files)
current_image = 0

# Batch size configuration
batch_size = 4  # Adjust based on your GPU memory capacity
image_size = (480, 480)  # Standardize image size to avoid dimension mismatches

print(f"Total images to process: {total_images}")
print(f"Batch size: {batch_size}")

# Process images in batches
for i in range(0, total_images, batch_size):
    batch_files = image_files[i:i + batch_size]
    batch_images = []
    batch_inputs = []
    batch_start_time = time.time()
    
    # Load and prepare images for the batch
    for img_file in batch_files:
        current_image += 1
        img_path = os.path.join(folder_path, img_file)
        
        try:
            image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format
            # Resize using OpenCV INTER_CUBIC
            image_np = np.array(image)
            image_resized = cv2.resize(image_np, image_size, interpolation=cv2.INTER_CUBIC)
            image = Image.fromarray(image_resized)
            batch_images.append((img_file, image))
        except Exception as e:
            print(f"[{current_image}/{total_images}] Unable to open image {img_file}: {e}")
            continue
        
        # Prepare conversation input for Qwen2.5-VL
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "List main objects in this image, one per line, without any additional attribute, description or explanation."}
                ]
            }
        ]
        
        # Convert conversation to text using apply_chat_template
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        # Process text and image together
        inputs = processor(
            text=[text],  # Wrap text in a list for batch processing
            images=[image],
            return_tensors="pt",
            padding=True,  # Enable padding for text and image tokens
            truncation=True  # Truncate sequences to max length
        ).to('cuda:0')
        
        batch_inputs.append((img_file, inputs))
    
    if not batch_inputs:
        print(f"[{i+1}-{i+len(batch_files)}/{total_images}] No valid images in batch, skipping...")
        continue
    
    # Combine inputs for batch processing
    try:
        batch_input_dict = {
            key: torch.cat([inputs[key] for _, inputs in batch_inputs], dim=0)
            if inputs[key].dim() > 1 else torch.stack([inputs[key] for _, inputs in batch_inputs], dim=0)
            for _, inputs in batch_inputs
            for key in inputs.keys()
        }
    except Exception as e:
        print(f"[{i+1}-{i+len(batch_files)}/{total_images}] Error combining inputs: {e}")
        continue
    
    # Generate responses for the batch
    try:
        generate_ids = model.generate(
            **batch_input_dict,
            max_new_tokens=500,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        results = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as e:
        print(f"[{i+1}-{i+len(batch_files)}/{total_images}] Error during model inference: {e}")
        continue
    
    # Process each result in the batch
    prompt = "List main objects in this image, one per line, without any additional attribute, description or explanation."
    for idx, (img_file, _) in enumerate(batch_inputs):
        try:
            result = results[idx]
            # Remove the prompt and unwanted dialogue artifacts
            description = re.sub(r'system\n.*?\nuser\n.*?assistant\n', '', result, flags=re.DOTALL)
            description = description.replace(prompt, "").strip()
            
            # Convert newline-separated objects to comma-separated
            categories = description.split('\n')
            categories = [cat.strip() for cat in categories if cat.strip()]  # Remove empty strings and extra whitespace
            description = ", ".join(categories)
            
            # Save output as JSON
            data = {
                "image": img_file,
                "description": description
            }
            
            json_file = os.path.splitext(img_file)[0] + '.json'
            json_path = os.path.join(out_dir, json_file)
            
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"[{current_image}/{total_images}] Failed to save {json_file}: {e}")
                continue
            
            print(f"[{current_image}/{total_images}] Processed {img_file}, Saved to {json_file}")
        
        except Exception as e:
            print(f"[{current_image}/{total_images}] Error processing result for {img_file}: {e}")
            continue
    
    elapsed_time = time.time() - batch_start_time
    print(f"Batch {i//batch_size + 1} completed, Time: {elapsed_time:.2f} seconds")

print("All images processed!")
