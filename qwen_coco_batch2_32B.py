import os
import json
import time
import torch
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Qwen2.5-VL model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto",
    attn_implementation="flash_attention_2"
)

# Load processor with fast tokenizer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", use_fast=True)
processor.tokenizer.padding_side = 'left'

# Define directories
json_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/qwen2.5_32B_matched_object_coco_mymachine_NN"
image_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/images/train2017"
output_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_descriptions_coco_NN"
os.makedirs(output_dir, exist_ok=True)

# Checkpoint log file
checkpoint_log = os.path.join(output_dir, "processed_files.log")

# Load or initialize processed files log, verify output existence
processed_files_set = set()
if os.path.exists(checkpoint_log):
    with open(checkpoint_log, 'r') as f:
        for line in f:
            line = line.strip()
            if line and os.path.exists(os.path.join(output_dir, line)):
                processed_files_set.add(line)
            else:
                print(f"Warning: {line} in processed_files.log but output file missing, will reprocess")
else:
    processed_files_set = set(f for f in os.listdir(output_dir) if f.endswith('.json') and f != 'processed_files.log')

# Get list of JSON files and filter out already processed ones
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json') and f not in processed_files_set]
total_files = len(json_files)
processed_files = len(processed_files_set)
print(f"Total JSON files to process: {total_files}") 
print(f"Already processed: {processed_files}")
print(f"Remaining files: {total_files}")

# Batch size configuration
batch_size = 1 # Adjust based on L40's 48GB VRAM
print(f"Batch size: {batch_size}")

# Function to load JSON and image
def load_json_and_image(json_file):
    json_path = os.path.join(json_dir, json_file)
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            image_name = data['image']
            object_classes = data['matched_objects']
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        return json_file, image_name, object_classes, image, None
    except Exception as e:
        return json_file, None, None, None, str(e)

# Async function to write JSON with retry
async def write_json_async(json_path, data, retries=3):
    for attempt in range(retries):
        try:
            async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=4))
            return True
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying write to {json_path} ({attempt + 1}/{retries}): {str(e)}")
                await asyncio.sleep(1)
            else:
                return str(e)

# Async function to log processed file with retry
async def log_processed_file(json_file, retries=3):
    for attempt in range(retries):
        try:
            async with aiofiles.open(checkpoint_log, 'a', encoding='utf-8') as f:
                await f.write(f"{json_file}\n")
            return True
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying log for {json_file} ({attempt + 1}/{retries}): {str(e)}")
                await asyncio.sleep(1)
            else:
                return str(e)

# Track total execution time
total_start_time = time.time()

# Process JSON files in batches
loop = asyncio.get_event_loop()
with ThreadPoolExecutor(max_workers=4) as executor:
    for i in range(0, total_files, batch_size):
        batch_files = json_files[i:i + batch_size]
        batch_start_time = time.time()
        batch_data = []
        batch_inputs = []
        batch_output_dicts = []

        # Parallel load JSON and images
        results = list(executor.map(load_json_and_image, batch_files))
        for json_file, image_name, object_classes, image, error in results:
            processed_files += 1
            if error:
                print(f"[{processed_files}/{total_files + len(processed_files_set)}] Error loading {json_file}: {error}")
                continue
            batch_data.append((json_file, image_name, object_classes, image))

        # Prepare inputs for batch
        for json_file, image_name, object_classes, image in batch_data:
            output_dict = {}
            text_inputs = []
            if not object_classes:  # Handle empty matched_objects explicitly
                print(f"[{processed_files}/{total_files + len(processed_files_set)}] {json_file} has empty matched_objects, saving empty output")
                batch_output_dicts.append((json_file, output_dict))
                continue
            for obj in object_classes:
                if not obj or obj.strip() == "":
                    print(f"[{processed_files}/{total_files + len(processed_files_set)}] Skipping {obj} in {json_file}: Category is empty or invalid")
                    output_dict[obj] = ["Skipped - Category is empty or invalid"]
                    continue
                # Build prompt for each object
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": (
                                    f"Describe the visual appearance of the {obj} if exists in the image in one detailed sentence "
                                    f"that strictly follows the structure 'The {obj}, is {{size}}, has a {{shape}} shape with a {{texture}} texture and in a {{color}} color', "
                                    f"including its specific size, shape, texture, and color as observed in the image, without skipping any of these aspects. "
                                    f"If the {obj} does not exist in the image, respond with 'The {obj} is not present in the image.'"
                                    f"The description must be under 77 tokens as per the CLIP tokenizer."
                                )
                            }
                        ]
                    }
                ]
                text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                text_inputs.append((obj, text))
            if text_inputs:
                batch_inputs.append((json_file, image_name, image, text_inputs, output_dict))
            else:
                batch_output_dicts.append((json_file, output_dict))

        # Process batch inputs
        if batch_inputs:
            try:
                all_texts = []
                all_images = []
                obj_mapping = []
                for json_file, image_name, image, text_inputs, output_dict in batch_inputs:
                    for obj, text in text_inputs:
                        all_texts.append(text)
                        all_images.append(image)
                        obj_mapping.append((json_file, image_name, obj, output_dict))

                # Process text and images together
                inputs = processor(
                    text=all_texts,
                    images=all_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to('cuda:0')

                # Generate descriptions
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=500,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )

                # Decode results
                results = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # Assign descriptions to output dictionaries
                for idx, (json_file, image_name, obj, output_dict) in enumerate(obj_mapping):
                    try:
                        description = results[idx].split("assistant\n")[-1].strip()
                        output_dict[obj] = [description]
                    except Exception as e:
                        print(f"[{processed_files}/{total_files + len(processed_files_set)}] Error processing {obj} in {json_file}: {e}")
                        output_dict[obj] = [f"Error - {str(e)}"]
                    if (json_file, output_dict) not in batch_output_dicts:
                        batch_output_dicts.append((json_file, output_dict))

            except Exception as e:
                print(f"[{i+1}-{i+len(batch_files)}/{total_files + len(processed_files_set)}] Error during batch inference: {e}")
                continue

        # Save outputs and log processed files asynchronously
        if batch_output_dicts:
            tasks = []
            for json_file, output_dict in batch_output_dicts:
                output_file = os.path.join(output_dir, json_file)
                tasks.append(write_json_async(output_file, output_dict))
                tasks.append(log_processed_file(json_file))
                print(f"[{processed_files}/{total_files + len(processed_files_set)}] Processed {json_file}, Saved to {output_file}")

            # Run async JSON writing and logging
            results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            for idx, result in enumerate(results):
                if isinstance(result, str):
                    print(f"[{processed_files}/{total_files + len(processed_files_set)}] Failed to save/log {batch_output_dicts[idx//2][0]}: {result}")
        else:
            print(f"[{i+1}-{i+len(batch_files)}/{total_files + len(processed_files_set)}] No outputs to save for batch, files: {', '.join(batch_files)}")

        elapsed_time = time.time() - batch_start_time
        print(f"Batch {i//batch_size + 1} completed, Time: {elapsed_time:.2f} seconds")

# Calculate and display total execution time
total_time = time.time() - total_start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")
print(f"Processed {processed_files} JSON files (including previously processed)")
