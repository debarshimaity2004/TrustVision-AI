import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import kagglehub

from model import DeepfakeResNet
from dataset import DeepfakeDataset

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("====================================")
    print(" Deepfake Detection Model Training ")
    print("====================================")
    
    # 1. Download Dataset
    print("\nDownloading dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
        print(f"Dataset securely downloaded/located at: {dataset_path}")
    except Exception as e:
        print(f"Failed to download dataset. Please install kagglehub: `pip install kagglehub`")
        print(f"Error: {e}")
        return

    # 2. Prepare DataLoader
    # ResNet expects 224x224 RGB images with specific ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Data augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nLoading dataset metadata...")
    # Creating Dataset instance
    dataset = DeepfakeDataset(data_dir=dataset_path, transform=transform)
    
    if len(dataset) == 0:
        print("Error: No valid images or videos found in the dataset with given heuristic labels.")
        return
        
    # Split into train and validation sets (80 / 20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Model setup
    print("\nConfiguring model...")
    model = DeepfakeResNet(pretrained=True).to(DEVICE)
    model_path = os.path.join("models", "model.pth")
    os.makedirs("models", exist_ok=True)
    
    # If weights exist, start from them
    if os.path.exists(model_path):
        print(f"Loading existing weights from {model_path} to resume training...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Using CrossEntropyLoss because we defined 2 output neurons in DeepfakeResNet
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    print("\nStarting Training...")
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # training loop
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': round(running_loss / (total/BATCH_SIZE + 1e-5), 4),
                'acc': round(100 * correct / total, 2)
            })
            
        train_acc = 100 * correct / total
        
        # validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val] ")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Summary - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"-> Validation accuracy improved. Saving model weights to {model_path}...")
            torch.save(model.state_dict(), model_path)

    print("\nTraining Complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
