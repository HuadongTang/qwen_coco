# qwen_coco
1.Dataset Pre

`wget http://images.cocodataset.org/zips/train2017.zip`

unzip train2017.zip

2.Set up environment

conda create -n edc_new python=3.9

source activate edc_new

pip install torch=='2.4.1+cu121' torchvision=='0.19.1+cu121' torchaudio=='2.4.1+cu121' --index-url https://download.pytorch.org/whl/cu121

pip install -U git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830

pip install pillow bitsandbytes accelerate huggingface_hub

pip install transformers==4.51.3 accelerate

pip install sentence_transformers

pip install nltk

pip install opencv-python

pip install -U flash-attn

pip install aiofiles


[3.Run](http://3.Run) the code

(1)先运行qwen_coco.py

```python
# Define input and output paths
folder_path = '/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/images/train2017'
out_dir = '/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_object_coco/'
```

需要设置folder_path 就是解压的coco训练数据的路径，输出的路径自己定义

(2)跑完再运行handle_caption.py,这里还是需要设置路径，input_folder就是刚刚跑完的out_dir送给现在的input_folder,输出路径自己定义

```python
# 输入输出路径
input_folder = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_object_coco/"
output_folder = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_coco/"
```

（3）最后运行qwen_coco2.py,同样需要设置路径，这里json_dir是我们handle_caption.py的输出out_folder的路径。其次image_dir是coco训练数据的路径，输出的我们自己定义。

```python
# Define directories
json_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_coco"  # Directory containing JSON files
image_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/images/train2017"  # Directory containing images
output_dir = "/projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_descriptions_coco"  # Directory to save output text files
```

运行完成后我需要这三个跑完的结果分别就是

1. /projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_object_coco/
2. /projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_coco/
3. /projects/huatang_proj/CAT-Seg-gpt/datasets/coco-stuff/qwen2.5_32B_matched_object_descriptions_coco

可以用压缩包的形式给我
