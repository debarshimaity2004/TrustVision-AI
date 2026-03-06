import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None, sample_limit=20000):
        """
        Deepfake Dataset Loader.
        Automatically scans the given directory for images
        and attempts to label them as REAL (0) or FAKE (1).
        
        Args:
            data_dir (str): Root dataset folder downloaded.
            transform: torchvision transforms.
            sample_limit (int): To limit dataset size for quick testing and memory limits. Default 20000.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sample_limit = sample_limit
        self.dataset_name = "xhlulu/140k-real-and-fake-faces"
        self.total_real = 0
        self.total_fake = 0
        self.samples = []
        
        # Build samples
        self._find_files()
        
        if self.sample_limit:
            # We want balanced classes:
            real_samples = [s for s in self.samples if s['label'] == 0]
            fake_samples = [s for s in self.samples if s['label'] == 1]
            
            random.shuffle(real_samples)
            random.shuffle(fake_samples)
            
            half_limit = self.sample_limit // 2
            
            self.samples = real_samples[:half_limit] + fake_samples[:half_limit]
            random.shuffle(self.samples)

    def _find_files(self):
        print(f"Scanning directory: {self.data_dir} for dataset files...")
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                ext = file.lower().split('.')[-1]
                path = os.path.join(root, file)
                path_lower = path.lower()
                
                # Assign labels
                # We check the direct parent folder name to prevent root directory names from polluting the label
                parent_dir = os.path.basename(root).lower()
                filename = file.lower()
                label = -1
                
                if any(x in parent_dir for x in ['real', 'original', 'youtube', 'pristine']):
                    label = 0
                elif any(x in parent_dir for x in ['fake', 'manipulated', 'df', 'face2face', 'fakes']):
                    label = 1
                else:
                    # Fallback to file name if folder doesn't have it
                    if any(x in filename for x in ['real', 'original']):
                        label = 0
                    elif any(x in filename for x in ['fake', 'manipulated']):
                        label = 1
                    
                if label != -1 and ext in ['jpg', 'jpeg', 'png', 'webp']:
                    self.samples.append({
                        "path": path,
                        "label": label,
                        "type": "image"
                    })
        
        print(f"Dataset scan complete. Found {len(self.samples)} valid samples.")
        self.total_real = sum(1 for s in self.samples if s['label'] == 0)
        self.total_fake = sum(1 for s in self.samples if s['label'] == 1)
        print(f"Class breakdown -> REAL: {self.total_real}, FAKE: {self.total_fake}")

    def __len__(self):
        return len(self.samples)



    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample['path']
        label = sample['label']
        
        img = None
        
        # We only work with images now from the new dataset.
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                
        # Fallback if corrupted file
        if img is None:
            img = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)
