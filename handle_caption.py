import os
import json
from sentence_transformers import SentenceTransformer, util
from nltk.stem import WordNetLemmatizer
import nltk
import re

# 下载 NLTK 模型
nltk.download('wordnet', quiet=True)

# 初始化模型和 lemmatizer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
lemmatizer = WordNetLemmatizer()

# 候选类别列表（COCO-Stuff 类别，添加 "container"）
candidate_class = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw", "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood"]

# 输入输出路径
input_folder = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_object_coco/"
output_folder = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_coco/"
os.makedirs(output_folder, exist_ok=True)

# 编码所有候选类别
candidate_embeddings = model.encode(candidate_class, convert_to_tensor=True, show_progress_bar=False)

# 初始化统计变量
total_files = 0
total_matched_objects = 0

# 处理每个文件
file_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".json")])
total_file_count = len(file_list)

for idx, filename in enumerate(file_list, 1):
    total_files += 1
    input_path = os.path.join(input_folder, filename)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[{idx}/{total_file_count}] 无法加载文件 {filename}: {e}")
        continue

    image_name = data.get("image", "")
    description = data.get("description", "")

    if not description:
        print(f"[{idx}/{total_file_count}] 文件 {filename} 的 description 字段为空")
        output_data = {"image": image_name, "matched_objects": []}
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        continue

    # 解析 description，处理逗号分隔的格式（"Zebra, Grass, Flowers, Shadow"）
    object_words = [obj.strip() for obj in description.split(',') if obj.strip()]
    object_words = [re.sub(r'[^\w\s-]', '', obj).strip() for obj in object_words]  # 去除标点，保留连字符

    if not object_words:
        print(f"[{idx}/{total_file_count}] 文件 {filename} 未提取到有效对象")
        output_data = {"image": image_name, "matched_objects": []}
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        continue

    # 计算相似性并匹配
    matched_objects = []
    for obj in object_words:
        # 对短语整体进行词形还原（仅对最后一个词进行单数化）
        words = obj.lower().split()
        if words:
            last_word = lemmatizer.lemmatize(words[-1], pos='n')
            lemma = ' '.join(words[:-1] + [last_word]) if len(words) > 1 else last_word
        else:
            continue

        obj_embedding = model.encode(lemma, convert_to_tensor=True)
        similarities = util.cos_sim(obj_embedding, candidate_embeddings)[0]
        max_sim = similarities.max().item()
        if max_sim > 0.5:
            max_sim_idx = similarities.argmax().item()
            matched_candidate = candidate_class[max_sim_idx]
            matched_objects.append(matched_candidate)
        else:
            print(f"[{idx}/{total_file_count}] 对象: {lemma}, 最大相似度: {max_sim:.3f}, 最接近: {candidate_class[similarities.argmax().item()]}")

    # 去重，优先保留单数形式
    normalized_objects = {}
    for obj in matched_objects:
        words = obj.split()
        lemma = lemmatizer.lemmatize(words[-1], pos='n') if words else obj
        if len(words) > 1:
            lemma = ' '.join(words[:-1] + [lemma])
        if lemma not in normalized_objects:
            normalized_objects[lemma] = obj
        else:
            if not normalized_objects[lemma].endswith('s') and obj.endswith('s'):
                continue
            elif normalized_objects[lemma].endswith('s') and not obj.endswith('s'):
                normalized_objects[lemma] = obj
            elif len(obj) < len(normalized_objects[lemma]):
                normalized_objects[lemma] = obj

    final_objects = list(normalized_objects.values())
    total_matched_objects += len(final_objects)

    # 打印处理信息
    print(f"[{idx}/{total_file_count}] 处理文件: {filename} | 提取对象: {object_words} | 匹配到: {final_objects}")

    # 保存输出
    output_data = {
        "image": image_name,
        "matched_objects": final_objects
    }
    output_path = os.path.join(output_folder, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
    except Exception as e:
        print(f"[{idx}/{total_file_count}] 无法保存文件 {output_path}: {e}")

# 打印最终统计信息
print(f"\n处理完成 ✅")
print(f"共处理文件数：{total_files}")
print(f"共识别出候选类别词数量（去重后求和）：{total_matched_objects}")
print(f"结果输出目录：{output_folder}")