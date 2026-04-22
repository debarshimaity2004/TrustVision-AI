import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import DeepfakeResNetViT
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
GRADCAM_DIR = os.path.join(OUTPUT_DIR, 'gradcam')
os.makedirs(GRADCAM_DIR, exist_ok=True)

MODELPATH = os.path.join(os.path.dirname(__file__), 'ml', 'models', 'model.pth')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def compute_landmark_score(image):
    try:
        import mediapipe as mp
    except ImportError:
        return 0.5

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = mp_face_mesh.process(np.array(image))
    if not results.multi_face_landmarks:
        return 0.5
    lm = results.multi_face_landmarks[0].landmark
    coords = np.array([[p.x, p.y] for p in lm])
    eye_left = coords[33]; eye_right = coords[263]; nose = coords[1]; mouth = coords[0]
    eye_dist = np.linalg.norm(eye_left - eye_right)
    nose_eye = np.linalg.norm(nose - (eye_left + eye_right) / 2)
    mouth_nose = np.linalg.norm(mouth - nose)
    if eye_dist == 0:
        return 0.5
    ratio1 = nose_eye / eye_dist
    ratio2 = mouth_nose / eye_dist
    ideal1, ideal2 = 0.35, 0.45
    score = 1.0 - (abs(ratio1 - ideal1) + abs(ratio2 - ideal2)) / 1.0
    score = np.clip((score + 1) / 2, 0.0, 1.0)
    return float(score)


def load_model(device):
    model = DeepfakeResNetViT(pretrained=False).to(device)
    checkpoint = torch.load(MODELPATH, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def infer_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    landmark_score = torch.tensor([compute_landmark_score(img)], dtype=torch.float32).to(device)

    output = model(inp, landmark_score)
    probs = torch.softmax(output, dim=1).cpu().squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    label = 'REAL' if pred_idx == 0 else 'FAKE'

    cam = GradCAM(model=model, target_layers=model.get_target_layer(), use_cuda=(device.type=='cuda'))
    grayscale_cam = cam(input_tensor=inp, eigen_smooth=True)[0]
    img_np = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    gradcam_path = os.path.join(GRADCAM_DIR, f"{base_name}_heatmap.jpg")
    Image.fromarray(cam_image).save(gradcam_path)

    confidence = float(probs[pred_idx] * 100.0)

    print("═══════════════════════════════")
    print(f"Image   : {os.path.basename(image_path)}")
    print(f"Result  : {label} ({confidence:.1f}% confident)")
    print(f"Heatmap : {gradcam_path}")
    print("═══════════════════════════════")

    return {
        'image': image_path,
        'result': label,
        'confidence': confidence,
        'heatmap': gradcam_path
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepfake detection demo')
    parser.add_argument('--image', required=True, help='Path to the image file')
    args = parser.parse_args()

    infer_image(args.image)
