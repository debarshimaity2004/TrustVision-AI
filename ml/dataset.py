import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None, sample_limit=None):
        """
        Deepfake Dataset Loader.
        Automatically scans the given directory for images or videos
        and attempts to label them as REAL (0) or FAKE (1).
        
        Args:
            data_dir (str): Root dataset folder downloaded.
            transform: torchvision transforms.
            sample_limit (int): To limit dataset size for quick testing.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Build samples
        self._find_files()
        
        if sample_limit:
            random.shuffle(self.samples)
            self.samples = self.samples[:sample_limit]

    def _find_files(self):
        print(f"Scanning directory: {self.data_dir} for dataset files...")
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                ext = file.lower().split('.')[-1]
                path = os.path.join(root, file)
                path_lower = path.lower()
                
                # Assign labels
                # We assume standard structure naming patterns
                label = -1
                if any(x in path_lower for x in ['real', 'original', 'youtube', 'pristine']):
                    label = 0
                elif any(x in path_lower for x in ['fake', 'manipulated', 'df', 'face2face', 'fakes']):
                    label = 1
                    
                if label != -1 and ext in ['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']:
                    self.samples.append({
                        "path": path,
                        "label": label,
                        "type": "video" if ext in ['mp4', 'avi', 'mov'] else "image"
                    })
        
        print(f"Dataset scan complete. Found {len(self.samples)} valid samples.")
        reals = sum(1 for s in self.samples if s['label'] == 0)
        fakes = sum(1 for s in self.samples if s['label'] == 1)
        print(f"Class breakdown -> REAL: {reals}, FAKE: {fakes}")

    def __len__(self):
        return len(self.samples)

    def _get_video_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            random_frame = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # Convert BGR to RGB
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample['path']
        label = sample['label']
        
        img = None
        if sample['type'] == 'video':
            img = self._get_video_frame(path)
        else:
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                
        # Fallback if corrupted file
        if img is None:
            img = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)
