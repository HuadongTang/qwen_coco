import os
import json
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image

# Configure 4-bit quantization to reduce memory usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # Double quantization for further memory optimization
)

# Load Qwen2.5-VL model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically distribute across available GPUs
)

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Define directories
json_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_coco"  # Directory containing JSON files
image_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/images/train2017"  # Directory containing images
output_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_descriptions_coco"  # Directory to save output text files

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Track total execution time
total_start_time = time.time()
processed_files = 0
total_files = len([f for f in os.listdir(json_dir) if f.endswith('.json')])

# Process each JSON file
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        processed_files += 1
        print(f"\nProcessing {processed_files}/{total_files}: {json_file}")
        start_time = time.time()
        
        # Read JSON file
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_name = data['image']
                object_classes = data['matched_objects']
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
        
        # Load image
        image_path = os.path.join(image_dir, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            continue

        # Prepare output dictionary
        output_dict = {}
        
        # Generate descriptions for each object class
        for obj in object_classes:
            # Check if the category is valid
            if not obj or obj.strip() == "":
                print(f"  Skipping {obj}: Category is empty or invalid")
                output_dict[obj] = ["Skipped - Category is empty or invalid"]
                continue

            print(f"  Generating description for {obj}...")
            
            # Process inputs for description
            try:
                # Build prompt text
                # Prepare conversation input for Qwen2.5-VL
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": 
                            f"Describe the visual appearance of the {obj} if exists in the image in one detailed sentence "
                            f"that strictly follows the structure 'The {obj}, is {{size}}, has a {{shape}} shape with a {{texture}} texture and in a {{color}} color', "
                            f"including its specific size, shape, texture, and color as observed in the image, without skipping any of these aspects. "
                            f"If the {obj} does not exist in the image, respond with 'The {obj} is not present in the image.'"}
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
                
                # Generate description
                description_ids = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

                # Decode the description
                description_result = processor.batch_decode(description_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                # Extract the description sentence or absence message
                description = description_result.strip()
                output_dict[obj] = [description]
                assistant_response = description_result.split("assistant\n")[-1].strip()
                output_dict[obj] = [assistant_response]
            except Exception as e:
                print(f"    Error processing {obj}: {e}")
                output_dict[obj] = [f"Error - {str(e)}"]
        
        # Save descriptions to output JSON file
        output_file = os.path.join(output_dir, json_file)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, ensure_ascii=False, indent=4)
            print(f"  Saved descriptions to {output_file}")
        except Exception as e:
            print(f"  Error saving descriptions for {json_file}: {e}")
        
        # Calculate and display processing time for this file
        file_time = time.time() - start_time
        print(f"  Time taken for {json_file}: {file_time:.2f} seconds")

# Calculate and display total execution time
total_time = time.time() - total_start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")
print(f"Processed {processed_files} JSON files")