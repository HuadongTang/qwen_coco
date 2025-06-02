import os
import json
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import re

# Configure 4-bit quantization to reduce memory usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
    # ,attn_implementation="flash_attention_2"
)

# Default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

# Define input and output paths
folder_path = '/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/images/train2017'
out_dir = '/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_object/'
os.makedirs(out_dir, exist_ok=True)

# Get list of valid image files and total count
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
total_images = len(image_files)
current_image = 0

print(f"Total images to process: {total_images}")

# Process each image
for img_file in image_files:
    current_image += 1
    img_path = os.path.join(folder_path, img_file)
    
    start_time = time.time()
    
    try:
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format
    except Exception as e:
        print(f"[{current_image}/{total_images}] Unable to open image {img_file}: {e}")
        continue
    
    # Prepare conversation input for Qwen2.5-VL
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "List main objects in this image, one per line, without any additional  attribute, description or explanation."}
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
        text=text,
        images=[image],
        return_tensors="pt"
    ).to('cuda:0')
    
    # Generate response
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Remove the prompt and unwanted dialogue artifacts
    prompt = "List main objects in this image, one per line, without any additional attribute, description or explanation."
    # Use regex to remove system/user/assistant dialogue parts
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
    
    elapsed_time = time.time() - start_time
    print(f"[{current_image}/{total_images}] Processed {img_file}, Time: {elapsed_time:.2f} seconds, Saved to {json_file}")

print("All images processed!")