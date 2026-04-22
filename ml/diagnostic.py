import os, sys, torch
from pathlib import Path
from model import DeepfakeResNetViT as DeepfakeModel
from dataset import DeepfakeDataset
from train import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE

print('--- DIAGNOSTIC START ---')

# 1. MODEL ARCHITECTURE
print('\n1. MODEL ARCHITECTURE')
model = DeepfakeModel(pretrained=False)
model_path = os.path.join('models', 'model.pth')
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        out = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('Loaded checkpoint with state_dict, strict=False')
        print('Missing keys:', out.missing_keys)
        print('Unexpected keys:', out.unexpected_keys)
    else:
        out = model.load_state_dict(checkpoint, strict=False)
        print('Loaded checkpoint directly as state_dict, strict=False')
        print('Missing keys:', out.missing_keys)
        print('Unexpected keys:', out.unexpected_keys)
else:
    print('WARNING: model.pth checkpoint not found; model is untrained random weights.')

print('Model summary:')
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters: {total_params}')
print(f'Trainable parameters: {trainable_params}')

model.eval()
with torch.no_grad():
    sample = torch.randn(1, 3, 224, 224)
    shapes = []
    def hook(m, inp, out):
        in_shape = tuple(inp[0].shape) if isinstance(inp, tuple) and len(inp)>0 and hasattr(inp[0], 'shape') else None
        out_shape = tuple(out.shape) if hasattr(out, 'shape') else None
        shapes.append((m.__class__.__name__, in_shape, out_shape))
    handles = []
    for n,m in model.named_modules():
        if not list(m.children()):
            handles.append(m.register_forward_hook(hook))
    out = model(sample)
    for h in handles: h.remove()

print('\nLayer shapes (leaf modules):')
for cls, inp, outp in shapes:
    print(f'{cls}: in={inp}, out={outp}')

# 2. DATASET INFO
print('\n2. DATASET INFO')
print('Label assignment logic from DeepfakeDataset:')
print('- exact folder name real => 0 (REAL)')
print('- exact folder name fake => 1 (FAKE)')
print('- OR CSV manifest with explicit path,label columns')
print('- no substring matching is used')
print('- image extensions jpg/jpeg/png/webp')
print('Class map: 0=REAL, 1=FAKE')

# attempt to locate dataset via explicit folder structure
found_path = None
for p in Path('..').rglob('*'):
    if p.is_dir() and p.name.lower() in ('real', 'fake'):
        found_path = p.parent
        break

if found_path:
    print(f'Found candidate dataset path: {found_path}')
    ds = DeepfakeDataset(str(found_path), transform=None, sample_limit=None)
    print(f'Dataset count: {len(ds)}')
    print(f'Class counts: real={ds.total_real}, fake={ds.total_fake}')
else:
    print('No dataset path found in repository, cannot compute actual class distribution train/val.')

print('\nTransforms applied in train.py:')
print('- Resize((224, 224))')
print('- RandomHorizontalFlip()')
print('- ColorJitter(brightness=0.1, contrast=0.1)')
print('- ToTensor()')
print('- Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])')

# 3. TRAINING CONFIG
print('\n3. TRAINING CONFIG')
print('Loss function: CrossEntropyLoss')
print(f'Optimizer: Adam, lr={LEARNING_RATE}')
print(f'Number of epochs configured: {NUM_EPOCHS}')
print('Class weights: None')

# 4. PREDICTIONS ON SAMPLE BATCH
print('\n4. PREDICTIONS ON SAMPLE BATCH')
if found_path:
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    ds = DeepfakeDataset(str(found_path), transform=tfms, sample_limit=512)
    n = len(ds)
    if n == 0:
        print('Dataset empty.')
    else:
        train_n = int(0.8*n);
        val_n = n-train_n
        val_ds = torch.utils.data.Subset(ds, list(range(train_n, n)))
        val_loader = DataLoader(val_ds, batch_size=10, shuffle=False)
        images, labels = next(iter(val_loader))
        model_cpu = model.cpu()
        model_cpu.eval()
        with torch.no_grad():
            logits = model_cpu(images)
        probs = torch.softmax(logits, dim=1)
        print('Raw logits first 10:')
        for i in range(min(10, logits.shape[0])):
            print(i, logits[i].tolist())
        print('Probabilities first 10:')
        for i in range(min(10, probs.shape[0])):
            print(i, probs[i].tolist())
        preds = torch.argmax(probs, dim=1)
        print('Pred vs Actual for first 10:')
        for i in range(min(10, len(labels))):
            print(i, f'pred={preds[i].item()} actual={labels[i].item()}')
        print('Index mapping: 0=REAL, 1=FAKE')

        # 5. CONFUSION MATRIX
        try:
            from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
            y_true=[]; y_pred=[]
            for imgs, labs in val_loader:
                with torch.no_grad():
                    out = model_cpu(imgs)
                preds2 = torch.argmax(out, dim=1)
                y_pred.extend(preds2.tolist()); y_true.extend(labs.tolist())
            print('\n5. CONFUSION MATRIX')
            print(confusion_matrix(y_true, y_pred))
            print('Classification report:')
            print(classification_report(y_true, y_pred, target_names=['REAL','FAKE']))
            print('Overall accuracy:', accuracy_score(y_true, y_pred))
        except Exception as e:
            print('sklearn not installed or error:', e)

        print('\n6. LOSS & ACCURACY HISTORY')
        print('No stored history in checkpoint; training script prints these during training only.')
else:
    print('Data not available; cannot run val_loader-based predictions/confusion matrix.')

print('\n--- DIAGNOSTIC END ---')
