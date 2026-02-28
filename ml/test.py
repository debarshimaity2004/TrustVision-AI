import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import kagglehub
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model import DeepfakeResNet
from dataset import DeepfakeDataset

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("====================================")
    print(" Deepfake Detection Model Testing ")
    print("====================================")
    
    # 1. Download/Verify Dataset Location
    print("\nEnsuring dataset is downloaded...")
    try:
        dataset_path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
        print(f"Dataset securely located at: {dataset_path}")
    except Exception as e:
        print(f"Failed to locate downloaded dataset. Is kagglehub installed? `pip install kagglehub`")
        print(f"Error: {e}")
        return

    # 2. DataLoader matching test behavior
    # For testing, we do not need data augmentation, just resize, to tensor, and normalize.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nLoading dataset metadata for testing...")
    dataset = DeepfakeDataset(data_dir=dataset_path, transform=transform)
    
    if len(dataset) == 0:
        print("Error: No valid test images or videos found in the dataset.")
        return
        
    # We will test on the full dataset or if you had separated train/test dirs, 
    # you would specify `data_dir` pointing to the test split.
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Model setup and loading weights
    model_path = os.path.join("models", "model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights at {model_path} not found. Train the model first using train.py.")
        return
        
    print("\nConfiguring model...")
    model = DeepfakeResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 4. Evaluation Loop
    print("\nStarting Evaluation...")
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # collect targets and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 5. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["REAL (0)", "FAKE (1)"])
    
    print("\n================== RESULTS ==================")
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report (Precision, Recall, F1):")
    print(report)
    print("=============================================")

if __name__ == "__main__":
    main()
