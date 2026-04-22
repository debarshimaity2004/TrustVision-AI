import os

model_path = os.path.join("models", "model.pth")
if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Deleted old checkpoint: {model_path}")
else:
    print(f"No checkpoint found at {model_path}")

print("Ready for fresh training with ResNet+ViT ensemble.")
