import os
# ==========================================
# 0. THE HIGH-SPEED DOWNLOAD HACK
# ==========================================
# This bypasses the throttled Hugging Face main server and downloads
# the 2.7GB model at maximum speed using an official mirror.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import glob
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ... (inside the loading section) ...
processor = AutoProcessor.from_pretrained("kkatiz/THAI-BLIP-2")

# ==========================================
# 1. SETUP & BATCHING
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

TEST_DIR = os.path.join("test", "test")
SAMPLE_SUB_FILE = "sample_submission.csv"
BATCH_SIZE = 8  # Optimal for 12GB VRAM (Processes 8 images simultaneously)

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        try:
            image = Image.open(img_path).convert("RGB")
            return img_id, image
        except Exception:
            return img_id, None

def collate_fn(batch):
    batch = [b for b in batch if b[1] is not None]
    ids = [b[0] for b in batch]
    images = [b[1] for b in batch]
    return ids, images

# ==========================================
# 2. LOAD THAI BLIP-2 (STANDARD, STABLE)
# ==========================================
print("1. Downloading & Loading THAI-BLIP-2 (Via High-Speed Mirror)...")

processor = AutoProcessor.from_pretrained("kkatiz/THAI-BLIP-2")

# We use float16 to perfectly fit the 2.7B parameters into 12GB VRAM.
# Because this is a standard Hugging Face model, there is no meta-device bug!
model = Blip2ForConditionalGeneration.from_pretrained(
    "kkatiz/THAI-BLIP-2",
    torch_dtype=torch.float16,
    use_safetensors=True,
    low_cpu_mem_usage=True
).to(DEVICE)

model.eval()

# ==========================================
# 3. FAST BATCHED INFERENCE
# ==========================================
print("\n2. Processing Test Images in Batches...")

# Case-insensitive search to catch .JPG, .jpeg, .PNG, etc.
test_images = []
for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
    test_images.extend(glob.glob(os.path.join(TEST_DIR, ext)))

test_images = list(set(test_images)) # Remove any accidental duplicates
print(f"Found {len(test_images)} images.")

dataset = ImageDataset(test_images)

# num_workers=0 ensures Windows doesn't crash from multiprocessing
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=0, 
    pin_memory=True
)

results = []

with torch.no_grad():
    for ids, images in tqdm(dataloader, desc="Captioning Batches"):
        if not images:
            continue
            
        # Process 8 images simultaneously
        inputs = processor(images=images, return_tensors="pt").to(DEVICE, torch.float16)
        
        # Generate 8 captions at once
        generated_ids = model.generate(**inputs, max_new_tokens=40, num_beams=4)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        for img_id, caption in zip(ids, captions):
            results.append({
                "image_id": img_id,
                "caption_pred": caption.strip()
            })
            
        # Manually clear the VRAM cache to prevent Out-Of-Memory errors
        del inputs, generated_ids
        torch.cuda.empty_cache()

# ==========================================
# 4. FORMAT SUBMISSION
# ==========================================
print("\n3. Formatting Submission...")

preds_df = pd.DataFrame(results)

sample_sub = pd.read_csv(SAMPLE_SUB_FILE)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)
preds_df["image_id"] = preds_df["image_id"].astype(str)

final_sub = pd.merge(
    sample_sub[["image_id"]],
    preds_df,
    on="image_id",
    how="left"
)

final_sub.rename(columns={"caption_pred": "caption"}, inplace=True)
final_sub["caption"].fillna("ภาพถ่ายของสิ่งของ", inplace=True)

final_sub.to_csv("submission_blip2_fast.csv", index=False)
print("SUCCESS! 'submission_blip2_fast.csv' is ready.")