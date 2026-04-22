import os
import sys
import platform
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import kagglehub
from torch.amp import autocast, GradScaler

from model import DeepfakeResNetViT as DeepfakeModel
from dataset import DeepfakeDataset

# Hyperparameters
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4
# Windows has issues with DataLoader workers, set to 0 on Windows
DEFAULT_NUM_WORKERS = 0 if platform.system() == "Windows" else 2
DEFAULT_SAMPLE_LIMIT = 20000
DEFAULT_USE_LANDMARKS = False
AMP_CHOICES = {"auto", "off", "float16", "bfloat16"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "gradcam"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


def load_dataset(data_path, sample_limit=DEFAULT_SAMPLE_LIMIT, use_landmarks=DEFAULT_USE_LANDMARKS):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DeepfakeDataset(
        data_dir=data_path,
        transform=transform,
        sample_limit=sample_limit,
        use_landmarks=use_landmarks,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after scanning labels.")
    return dataset


def parse_dataset_source(argv):
    config = {
        "dataset_path": None,
        "resume": False,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": DEFAULT_NUM_EPOCHS,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "num_workers": DEFAULT_NUM_WORKERS,
        "sample_limit": DEFAULT_SAMPLE_LIMIT,
        "use_landmarks": DEFAULT_USE_LANDMARKS,
        "amp_mode": "auto",
    }

    for arg in argv[1:]:
        if arg.startswith("--data="):
            config["dataset_path"] = arg.split("=", 1)[1]
        elif arg.startswith("--manifest="):
            config["dataset_path"] = arg.split("=", 1)[1]
        elif arg == "--resume":
            config["resume"] = True
        elif arg.startswith("--batch-size="):
            config["batch_size"] = max(1, int(arg.split("=", 1)[1]))
        elif arg.startswith("--epochs="):
            config["epochs"] = max(1, int(arg.split("=", 1)[1]))
        elif arg.startswith("--lr="):
            config["learning_rate"] = float(arg.split("=", 1)[1])
        elif arg.startswith("--num-workers="):
            config["num_workers"] = max(0, int(arg.split("=", 1)[1]))
        elif arg.startswith("--workers="):
            config["num_workers"] = max(0, int(arg.split("=", 1)[1]))
        elif arg.startswith("--sample-limit="):
            parsed_limit = int(arg.split("=", 1)[1])
            config["sample_limit"] = None if parsed_limit <= 0 else parsed_limit
        elif arg == "--full-dataset":
            config["sample_limit"] = None
        elif arg == "--use-landmarks":
            config["use_landmarks"] = True
        elif arg == "--no-landmarks":
            config["use_landmarks"] = False
        elif arg.startswith("--amp="):
            amp_mode = arg.split("=", 1)[1].strip().lower()
            if amp_mode not in AMP_CHOICES:
                supported = ", ".join(sorted(AMP_CHOICES))
                raise ValueError(f"Unsupported --amp value `{amp_mode}`. Use one of: {supported}")
            config["amp_mode"] = amp_mode

    return config


def load_checkpoint(model, model_path, device):
    if not os.path.exists(model_path):
        return None
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    normalized = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(normalized, strict=True)
    return checkpoint


def resolve_amp_mode(amp_mode, device):
    if device.type != "cuda" or amp_mode == "off":
        return False, None, "off"

    if amp_mode == "float16":
        return True, torch.float16, "float16"

    if amp_mode == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, "bfloat16"
        print("Requested --amp=bfloat16, but this GPU does not support it. Falling back to full precision.")
        return False, None, "off"

    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, "bfloat16"

    # This model is considerably less stable under fp16 autocast on older GPUs.
    return False, None, "off"


def get_autocast_context(device, amp_enabled, amp_dtype):
    if device.type != "cuda" or not amp_enabled:
        return nullcontext()
    return autocast(device_type="cuda", dtype=amp_dtype)


def build_loader(dataset, batch_size, shuffle, num_workers):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": (DEVICE.type == "cuda"),
        "drop_last": shuffle,  # Prevent 1-sized batches causing BatchNorm errors during training
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def format_precision_hint(batch_size):
    safer_batch_size = max(1, batch_size // 2)
    return f"Retry with `--amp=off --batch-size={safer_batch_size}` or `--amp=bfloat16` if your GPU supports it."


def move_batch_to_device(inputs, labels, lmarks, device):
    non_blocking = device.type == "cuda"
    return (
        inputs.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
        lmarks.to(device, non_blocking=non_blocking),
    )


def evaluate(model, loader, device):
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

    model.eval()
    preds = []
    true = []
    probs = []
    filenames = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, labels, lmarks = batch
            else:
                inputs, labels = batch
                lmarks = torch.full((inputs.size(0),), 0.5)

            inputs, labels = inputs.to(device), labels.to(device)
            lmarks = lmarks.to(device)

            outputs = model(inputs, lmarks)
            soft = torch.softmax(outputs, dim=1)
            _, pred = torch.max(soft, 1)

            preds.extend(pred.cpu().tolist())
            true.extend(labels.cpu().tolist())
            probs.extend(soft[:, 1].cpu().tolist())

    cm = confusion_matrix(true, preds, labels=[0, 1])
    report = classification_report(true, preds, target_names=["REAL", "FAKE"], zero_division=0)
    acc = accuracy_score(true, preds)
    try:
        auc = roc_auc_score(true, probs)
    except Exception:
        auc = float("nan")

    out_file = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(out_file, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")

    print("--- EVALUATION ---")
    print(cm)
    print(report)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")

    return cm, report, acc, auc


def plot_history(train_losses, val_losses, train_accs, val_accs):
    import matplotlib.pyplot as plt
    import seaborn as sns

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=train_losses, label='Train Loss')
    sns.lineplot(x=epochs, y=val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'training_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=train_accs, label='Train Acc')
    sns.lineplot(x=epochs, y=val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'training_accuracy_curve.png'))
    plt.close()


def main():
    ensure_output_dirs()

    print("====================================")
    print(" Deepfake Detection Model Training ")
    print("====================================")

    try:
        config = parse_dataset_source(sys.argv)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return

    dataset_path = config["dataset_path"]
    resume_training = config["resume"]
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    num_workers = config["num_workers"]
    sample_limit = config["sample_limit"]
    use_landmarks = config["use_landmarks"]

    if dataset_path is None:
        print("\nDownloading dataset from Kaggle...")
        try:
            dataset_path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
            print(f"Dataset securely downloaded/located at: {dataset_path}")
        except Exception as e:
            print(f"Failed to download dataset. Please install kagglehub: `pip install kagglehub`")
            print(f"Error: {e}")
            return

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return

    print(f"Dataset path: {dataset_path}")

    try:
        dataset = load_dataset(
            dataset_path,
            sample_limit=sample_limit,
            use_landmarks=use_landmarks,
        )
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return

    print(f"Dataset Name: {dataset.dataset_name}")
    print(f"Label Source: {dataset.label_source}")
    print(f"Total RAW Real Images: {dataset.total_real}")
    print(f"Total RAW Fake Images: {dataset.total_fake}")
    print(f"Active Training Set: {len(dataset)} images")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Workers: {num_workers}")
    print(f"Landmark Scores: {'enabled' if use_landmarks else 'disabled'}")

    if dataset.total_real == 0 or dataset.total_fake == 0:
        print("ERROR: dataset has only one class; cannot train.")
        return

    total_count = dataset.total_real + dataset.total_fake
    weights = torch.tensor(
        [total_count / (2.0 * dataset.total_real), total_count / (2.0 * dataset.total_fake)],
        device=DEVICE,
    )

    # split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = DeepfakeModel(pretrained=True).to(DEVICE)
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if DEVICE.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Note: disabling benchmark prevents random cuDNN crashes due to varying memory mapping across batches
        torch.backends.cudnn.benchmark = False
    else:
        print("Running on CPU.")

    if resume_training:
        try:
            checkpoint = load_checkpoint(model, model_path, DEVICE)
            if checkpoint:
                print("Checkpoint loaded successfully (strict key match).")
        except Exception as e:
            print(f"Could not load checkpoint strictly: {e}")
            print("Starting from scratch.")
    elif os.path.exists(model_path):
        print(f"Existing checkpoint found at {model_path}, but training will start from scratch.")
        print("Pass --resume if you explicitly want to continue from the saved checkpoint.")

    amp_enabled, amp_dtype, resolved_amp_mode = resolve_amp_mode(config["amp_mode"], DEVICE)
    print(f"AMP Mode: {resolved_amp_mode}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0

    scaler = GradScaler(enabled=(DEVICE.type == "cuda" and amp_enabled and amp_dtype == torch.float16))

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for inputs, labels, lmarks in train_bar:
            inputs, labels, lmarks = move_batch_to_device(inputs, labels, lmarks, DEVICE)

            optimizer.zero_grad(set_to_none=True)

            try:
                with get_autocast_context(DEVICE, amp_enabled, amp_dtype):
                    outputs = model(inputs, lmarks)
                    loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print(f"\nWarning: NaN loss detected at epoch {epoch}, skipping batch.")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as exc:
                message = str(exc)
                if DEVICE.type == "cuda" and ("CUBLAS_STATUS" in message or "CUDA error" in message or "cuDNN" in message):
                    raise RuntimeError(f"{message}\n{format_precision_hint(batch_size)}") from exc
                raise

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'loss': round((running_loss / total), 4),
                'acc': round(100 * correct / total, 2)
            })

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
            for inputs, labels, lmarks in val_bar:
                inputs, labels, lmarks = move_batch_to_device(inputs, labels, lmarks, DEVICE)

                with get_autocast_context(DEVICE, amp_enabled, amp_dtype):
                    outputs = model(inputs, lmarks)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch}/{num_epochs}] - Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}% | Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"-> New best val acc. Saving checkpoint to {model_path}")
            torch.save({
                'state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'class_map': {'REAL': 0, 'FAKE': 1},
                'label_source': dataset.label_source
            }, model_path)

    print("\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.2f}%")

    # evaluation
    cm, report, acc, auc = evaluate(model, val_loader, DEVICE)
    plot_history(train_losses, val_losses, train_accs, val_accs)

    # save confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")


if __name__ == '__main__':
    main()
