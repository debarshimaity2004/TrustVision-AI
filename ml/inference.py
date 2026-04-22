import torch
import cv2
import numpy as np
import base64
from torchvision import transforms
from PIL import Image
from scipy.fftpack import dct
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from .model import DeepfakeResNetViT as DeepfakeResNet

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class DeepfakeDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # Construct the absolute path so we can call inference.py from anywhere (like the Backend root)
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.pth")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the architecture
        self.model = DeepfakeResNet(pretrained=False).to(self.device)
        self.model.eval()
        
        # Ensure the model directory exists
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else 'models', exist_ok=True)
        
        # Load weights if available, else use randomly initialized network (useful for verifying pipeline)
        self.class_map = {'REAL': 0, 'FAKE': 1}  # fallback default

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                self.class_map = self._normalize_class_map(checkpoint.get('class_map', self.class_map))
                print(f"Loaded trained model checkpoint from {model_path} with class_map={self.class_map}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded trained model weights from {model_path} (no class_map metadata). Using default class_map={self.class_map}")
        else:
            print(f"Warning: Model weights {model_path} not found. Running with random initialization.")

        # Data transformations required for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize Grad-CAM visualization
        self.cam = GradCAM(model=self.model, target_layers=self.model.get_target_layer())
        
        # Initialize OpenCV Face Detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) if MEDIAPIPE_AVAILABLE else None

    def _normalize_class_map(self, class_map):
        if not isinstance(class_map, dict):
            return {'REAL': 0, 'FAKE': 1}

        if 'REAL' in class_map and 'FAKE' in class_map:
            return {'REAL': int(class_map['REAL']), 'FAKE': int(class_map['FAKE'])}

        if 0 in class_map and 1 in class_map:
            normalized = {str(v).upper(): int(k) for k, v in class_map.items()}
            if 'REAL' in normalized and 'FAKE' in normalized:
                return {'REAL': normalized['REAL'], 'FAKE': normalized['FAKE']}

        return {'REAL': 0, 'FAKE': 1}

    def _compute_landmark_score(self, pil_image):
        if self.mp_face_mesh is None:
            return 0.5

        image_np = np.array(pil_image)
        results = self.mp_face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            return 0.5

        face_landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y] for lm in face_landmarks])

        eye_left = coords[33]
        eye_right = coords[263]
        nose = coords[1]
        mouth = coords[0]
        eye_dist = np.linalg.norm(eye_left - eye_right)
        nose_eye = np.linalg.norm(nose - (eye_left + eye_right) / 2)
        mouth_nose = np.linalg.norm(mouth - nose)

        if eye_dist == 0 or nose_eye == 0 or mouth_nose == 0:
            return 0.5

        ratio1 = nose_eye / eye_dist
        ratio2 = mouth_nose / eye_dist
        ideal1, ideal2 = 0.35, 0.45
        score = 1.0 - (abs(ratio1 - ideal1) + abs(ratio2 - ideal2)) / 1.0
        score = np.clip((score + 1) / 2, 0.0, 1.0)
        return float(score)

    def _predict_probs(self, pil_img):
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device, non_blocking=True)
        landmark_score = torch.tensor([self._compute_landmark_score(pil_img)], dtype=torch.float32, device=self.device)

        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            output = self.model(input_tensor, landmark_score)
            probs = torch.nn.functional.softmax(output, dim=1)[0]

        return input_tensor, probs

    def detect_frequency_artifacts(self, gray_img):
        """
        State-of-the-Art Generative Models (Stable Diffusion, Midjourney) 
        upsample images via decoders that leave unnatural repeating frequencies 
        (checkerboard artifacts) in the high-frequency spectrum.
        """
        # Compute 2D Discrete Cosine Transform
        dct_y = dct(dct(gray_img.T, norm='ortho').T, norm='ortho')
        
        # Calculate power spectrum
        power_spectrum = np.abs(dct_y) ** 2
        
        # Isolate High Frequency vs Low Frequency domains
        h, w = power_spectrum.shape
        high_freq_energy = np.sum(power_spectrum[int(h*0.5):, int(w*0.5):])
        total_energy = np.sum(power_spectrum)
        
        if total_energy == 0:
            return 0
            
        ratio = high_freq_energy / total_energy
        return ratio

    def detect_face(self, img_bgr):
        """Detects the largest face in an image and crops it with a margin."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # detectMultiScale(image, scaleFactor, minNeighbors)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
            
        # Select the largest face by bounding box area (w * h)
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Add a 20% margin to the bounding box
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_bgr.shape[1], x + w + margin_x)
        y2 = min(img_bgr.shape[0], y + h + margin_y)
        
        face_img = img_bgr[y1:y2, x1:x2]
        return face_img, (x1, y1, x2, y2)

    def predict_image(self, image_bytes: bytes):
        """
        Accepts raw image bytes, runs face detection, deepfake inference,
        and generates Grad-CAM heatmap overlay.
        """
        try:
            # Decode bytes to OpenCV BGR image
            np_img = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                raise ValueError("Could not decode image bytes.")

            # Detect and crop face
            face_img, face_box = self.detect_face(img_bgr)
            
            # If no face is found, process the entire image as fallback
            if face_img is None:
                face_img = img_bgr
                face_box = (0, 0, img_bgr.shape[1], img_bgr.shape[0])

            # Convert BGR (OpenCV) to RGB (Model/Pillow)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            full_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            full_pil_img = Image.fromarray(full_rgb)

            # Training used uncropped dataset images with landmark scores.
            # Blend full-image and face-crop probabilities at inference to reduce deployment shift.
            input_tensor, face_probs = self._predict_probs(pil_img)
            _, full_probs = self._predict_probs(full_pil_img)
            probs = (0.6 * face_probs) + (0.4 * full_probs)

            # Resolve class indices from class_map (checkpoint metadata) to prevent reversal
            real_index = int(self.class_map.get('REAL', 0))
            fake_index = int(self.class_map.get('FAKE', 1))
            real_prob = probs[real_index].item()
            fake_prob = probs[fake_index].item()
            
            # --- V2 Heuristic: AI Smoothness & Frequency Domain Artifacts ---
            # Modern Diffusion models (like Midjourney) output unnaturally smooth gradients.
            # They also leave statistical compression footprints in the high-frequency spectrum.
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # 1. Analyze Mathematical DCT spectrum for Generative Checkerboard Artifacts
            hf_ratio = self.detect_frequency_artifacts(gray_face)
            
            # 2. Analyze Canny Edge Density
            # Real human faces have pores, fine hairs, and subsurface light scattering that creates micro-edges.
            # AI generators output plastic-like skin that dramatically reduces edge density.
            edges = cv2.Canny(gray_face, 100, 200)
            edge_density = np.sum(edges) / (gray_face.shape[0]*gray_face.shape[1])
            
            # Heuristic Penalty Voting System
            # We don't want to accidentally flag real photographes taken with Soft Focus / Bokeh lenses
            penalty_score = 0
            if laplacian_var < 150: penalty_score += 1     # Fails Smoothness / Film Grain check
            if hf_ratio < 0.0001: penalty_score += 1       # Fails Mathematical DCT upscale check
            if edge_density < 8: penalty_score += 1        # Fails physical micro-edge texture check
            
            # If the image fails at least 2 out of 3 physics/reality checks simultaneously:
            # DISABLED: This heuristic is causing false positives on real images
            # if penalty_score >= 2:
            #     # Boost the fake probability by simulating a domain-shift correction
            #     fake_prob += 4.0  
                
            # Normalize probabilities
            total = real_prob + fake_prob
            fake_prob = fake_prob / total
            real_prob = real_prob / total
            
            authenticity_score = real_prob * 100

            pred_index = real_index if real_prob >= fake_prob else fake_index
            prediction = "REAL" if pred_index == real_index else "FAKE"
            confidence = max(real_prob, fake_prob) * 100
            
            # Determine Risk Level
            if prediction == "FAKE" and confidence > 80:
                risk_level = "HIGH"
            elif prediction == "FAKE" or (prediction == "REAL" and confidence < 70):
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
                
            # 2. Grad-CAM Interpretation
            # GradCAM requires gradients, so we run it outside torch.no_grad()
            # We visualize the class with the highest probability (targets=None defaults to max class)
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
            
            # Resize the original RGB face crop to match the heatmap size (224x224) 
            # and normalize to [0,1] for overlaying
            resized_face = cv2.resize(face_rgb, (224, 224)) / 255.0
            
            # Overlay heatmap using jet colormap
            cam_image = show_cam_on_image(resized_face, grayscale_cam, use_rgb=True)
            
            # Convert back to BGR to encode as Base64 format
            cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', cam_image_bgr)
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "success": True,
                "authenticity_score": round(authenticity_score, 3),
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "risk_level": risk_level,
                "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}",
                "face_box": {"x1": face_box[0], "y1": face_box[1], "x2": face_box[2], "y2": face_box[3]}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def predict_video(self, video_path: str, fps_sample_rate=1):
        """
        Extracts frames from a video path, samples 'fps_sample_rate' frames per second,
        runs image inference on each frame, and aggregates the predictions.
        (Note: Since fastapi handles video uploads by writing them temporarily,
        accepting a file path is usually best for video processing with cv2).
        """
        try:
            # OpenCV VideoCapture handles video files easily
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")
                
            # Grab original fps
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine step size to get fps_sample_rate frames per second
            if fps <= 0: fps = 30 # fallback
            frame_step = int(max(1, fps / fps_sample_rate))
            
            frame_scores = []
            frame_predictions = []
            frame_confidences = []
            frame_fake_probs = []
            heatmaps = []

            current_frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process only frames based on step logic
                if current_frame_idx % frame_step == 0:
                    # Encode frame back to bytes so we can use existing predict_image logic
                    # (In a highly optimized system we would extract this directly, but for now reuse is best)
                    success, buffer = cv2.imencode('.jpg', frame)
                    if success:
                        img_bytes = buffer.tobytes()
                        result = self.predict_image(img_bytes)
                        
                        if result.get("success"):
                            frame_scores.append(result["authenticity_score"])
                            frame_predictions.append(result["prediction"])
                            frame_confidences.append(result["confidence"])
                            frame_fake_probs.append(max(0.0, 1.0 - (result["authenticity_score"] / 100.0)))
                            # Maybe we just want one representative heatmap or all of them
                            heatmaps.append(result["heatmap_base64"])
                    
                current_frame_idx += 1

            cap.release()
            
            if not frame_scores:
                raise ValueError("Could not process any frames from the video.")

            # Aggregate results using simple averaging
            avg_score = sum(frame_scores) / len(frame_scores)
            avg_confidence = sum(frame_confidences) / len(frame_confidences)
            avg_fake_prob = sum(frame_fake_probs) / len(frame_fake_probs)
            
            final_prediction = "FAKE" if avg_fake_prob > 0.5 else "REAL"
            
            if final_prediction == "FAKE" and avg_confidence > 80:
                risk_level = "HIGH"
            elif final_prediction == "FAKE" or (final_prediction == "REAL" and avg_confidence < 70):
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                "success": True,
                "authenticity_score": round(avg_score, 2),
                "prediction": final_prediction,
                "confidence": round(avg_confidence, 2),
                "risk_level": risk_level,
                "total_frames_analyzed": len(frame_scores),
                # Return the very first heatmap as a thumbnail for the report/frontend
                "heatmap_base64_thumbnail": heatmaps[0] if heatmaps else None 
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
