import os
from PIL import Image
import io

def save_result(image, prompt, save_dir, epoch, method, batch_idx, sample_idx):
    """
    
    Args:
        image: PIL 
        prompt: 
        save_dir: 
        epoch: 
        method:  ('ori'  'ours')
        batch_idx: 
        sample_idx: 
    """
    # 
    method_dir = os.path.join(save_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # 
    epoch_dir = os.path.join(method_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 
    file_prefix = f"sample_{batch_idx}_{sample_idx}"
    
    # 
    image_path = os.path.join(epoch_dir, f"{file_prefix}.png")
    image.save(image_path)
    
    # 
    prompt_path = os.path.join(epoch_dir, f"{file_prefix}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

def save_batch_results(images, prompts, save_dir, epoch, method, batch_idx):
    """
    
    Args:
        images: 
        prompts: 
        save_dir: 
        epoch: 
        method:  ('ori'  'ours')
        batch_idx: 
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for sample_idx, (image, prompt) in enumerate(zip(images, prompts)):
        save_result(image, prompt, save_dir, epoch, method, batch_idx, sample_idx)