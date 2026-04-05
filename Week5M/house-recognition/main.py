import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------
# 1. Custom Dataset Class 
# ---------------------------------------------------------
class HouseDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_col = 'id' if self.is_test else 'image_name'
        img_name = str(self.df.iloc[idx][name_col]) 
        
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(self.img_dir, img_name + '.jpg')
        else:
            img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, img_name 
        
        label = float(self.df.iloc[idx]['class'])
        return image, label

# ---------------------------------------------------------
# 2. Main Execution Block (UPGRADED)
# ---------------------------------------------------------
if __name__ == '__main__':
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # --- UPGRADED HYPERPARAMETERS ---
    BATCH_SIZE = 16       # Lowered to fit larger images in 12GB VRAM
    EPOCHS = 15           # Increased to allow the scheduler to work
    LEARNING_RATE = 3e-4  # Slightly higher starting LR
    IMG_SIZE = 384        # Increased from 224 to 384 for higher fidelity

    TRAIN_DIR = os.path.join('train', 'train')
    TEST_DIR = os.path.join('test', 'test')

    # --- UPGRADED AUGMENTATIONS ---
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Handle lighting differences
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Simulate camera shifts
        transforms.ToTensor(),
        normalize
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    # --- Data Filtering ---
    print(f"Verifying which files actually exist in the '{TRAIN_DIR}' folder...")
    train_df = pd.read_csv('train.csv')
    
    valid_rows = []
    for index, row in train_df.iterrows():
        img_name = str(row['image_name'])
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(TRAIN_DIR, img_name + '.jpg')
        else:
            img_path = os.path.join(TRAIN_DIR, img_name)
            
        if os.path.exists(img_path):
            valid_rows.append(row)
            
    train_df = pd.DataFrame(valid_rows)
    print(f"Found {len(train_df)} valid images out of the original CSV.")

    # --- Load Data & Create DataLoaders ---
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['class'])

    train_dataset = HouseDataset(train_data, img_dir=TRAIN_DIR, transform=train_transforms)
    val_dataset = HouseDataset(val_data, img_dir=TRAIN_DIR, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- UPGRADED MODEL (EfficientNet-B3) ---
    print("Loading EfficientNet-B3...")
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)

    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- ADDED LEARNING RATE SCHEDULER ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Training Loop ---
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=correct_train/total_train)

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # Validation Phase
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = correct_val / total_val
        print(f"Validation Accuracy: {val_acc:.4f} | Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_house_model_v2.pth')
            print("--> Saved new best model!")

    # --- Inference & Submission ---
    print("\nTraining complete. Generating predictions for test set...")

    model.load_state_dict(torch.load('best_house_model_v2.pth'))
    model.eval()

    sub_df = pd.read_csv('sample_submission.csv')
    test_dataset = HouseDataset(sub_df, img_dir=TEST_DIR, transform=val_test_transforms, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Predicting"):
            images = images.to(DEVICE)
            outputs = model(images)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy().flatten()
            
            predictions.extend(preds)
            
            cleaned_names = [name.replace('.jpg', '').replace('.png', '') for name in img_names]
            image_ids.extend(cleaned_names)

    final_submission = pd.DataFrame({
        'id': image_ids,
        'answer': predictions
    })

    final_submission.to_csv('my_submission_v2.csv', index=False)