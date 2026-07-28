import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip

class SmallDataset(Dataset):
    def __init__(self, captions_file, img_dir, preprocess, tokenizer):
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        with open(captions_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        self.data = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                self.data.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)
        except Exception:
            image = torch.zeros((3, 224, 224)) # Black image fallback
            
        text = self.tokenizer(caption)[0]
        return image, text

def main():
    # Force CPU training (Kyunki tere paas Nvidia GPU nahi hai)
    device = "cpu"
    print(f"Training strictly on: {device}")

    # Chhota aur fast model load karo (ViT-B-32 heavy ho sakta hai CPU ke liye)
    print("Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', force_quick_gelu=True
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    dataset = SmallDataset("data/cc3m/captions.txt", "data/cc3m/images", preprocess, tokenizer)
    
    # Batch size chhota rakhna zaroori hai (4 ya 8) taaki 16GB RAM bhar na jaye
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=5e-6) # Learning rate aur kam kiya
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 1 # CPU pe 1 epoch hi bohot time lega!
    
    print("Starting training... (Grab a coffee, this will take time on a CPU)")
    model.train()
    
    for epoch in range(num_epochs):
        for batch_idx, (images, texts) in enumerate(dataloader):
            images, texts = images.to(device), texts.to(device)
            
            optimizer.zero_grad()
            
            image_features, text_features, logit_scale = model(images, texts)
            
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T
            
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            
            loss = (loss_fn(logits_per_image, ground_truth) + loss_fn(logits_per_text, ground_truth)) / 2
            
            loss.backward()
            optimizer.step()
            
            print(f"Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")

    # Model Save
    torch.save(model.state_dict(), "my_finetuned_clip.pt")
    print("Training Done! Saved as my_finetuned_clip.pt")

if __name__ == "__main__":
    main()