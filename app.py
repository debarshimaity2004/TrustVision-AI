import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model import DeepfakeResNetViT
from demo import compute_landmark_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import gradio as gr

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'models', 'model.pth')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
GRADCAM_DIR = os.path.join(OUTPUT_DIR, 'gradcam')
os.makedirs(GRADCAM_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeResNetViT(pretrained=False).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


MODEL, DEVICE = load_model()


def predict_image(image):
    img = image.convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    landmark_score = torch.tensor([compute_landmark_score(img)], dtype=torch.float32, device=DEVICE)

    out = MODEL(input_tensor, landmark_score)
    proba = torch.softmax(out, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(proba))
    label_text = 'REAL' if pred_idx == 0 else 'FAKE'
    confidence = float(proba[pred_idx] * 100.0)

    cam = GradCAM(model=MODEL, target_layers=MODEL.get_target_layer(), use_cuda=(DEVICE.type == 'cuda'))
    grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True)[0]
    img_np = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    heatmap_path = os.path.join(GRADCAM_DIR, 'latest_heatmap.jpg')
    Image.fromarray(cam_image).save(heatmap_path)

    output_text = f"Prediction: {label_text} ({confidence:.1f}% confidence)"
    return output_text, img, Image.fromarray(cam_image)


def demo():
    with gr.Blocks(title='Deepfake Detection Demo') as block:
        gr.Markdown('# Deepfake Detection System')
        with gr.Row():
            img_in = gr.Image(type='pil', label='Upload Image')
            with gr.Column():
                result = gr.Textbox(label='Prediction')
                heatmap_img = gr.Image(label='GradCAM Heatmap')
        img_in.change(predict_image, inputs=img_in, outputs=[result, gr.Image.update(), heatmap_img])
    block.launch()


if __name__ == '__main__':
    demo()
