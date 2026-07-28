from datasets import load_dataset
from PIL import Image
import os

print("Downloading dataset sample...")
# Sirf 1000 images download karenge taaki CPU jhel sake
dataset = load_dataset("pixparse/cc3m-wds", split="train", streaming=True)

# Folders banao
os.makedirs("data/cc3m/images", exist_ok=True)

captions = []
count = 0

for sample in dataset:
    try:
        image = sample["jpg"]
        caption = sample["txt"]
        
        # Image save karo
        image_name = f"{count}.jpg"
        image.save(f"data/cc3m/images/{image_name}")
        
        # Caption save karo
        captions.append(f"{image_name}\t{caption}")
        count += 1
        
        if count % 100 == 0:
            print(f"Downloaded {count} images...")
            
        if count >= 1000: # Limit to 1000 images for CPU
            break
    except Exception as e:
        continue

with open("data/cc3m/captions.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(captions))
    
print("Download complete! 1000 images and captions saved.")