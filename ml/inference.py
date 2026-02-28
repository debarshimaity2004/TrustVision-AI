import torch
import cv2
import numpy as np
import base64
from torchvision import transforms
from PIL import Image
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from .model import DeepfakeResNet

class DeepfakeDetector:
    def __init__(self, model_path="models/model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the architecture
        self.model = DeepfakeResNet(pretrained=False).to(self.device)
        self.model.eval()
        
        # Ensure the model directory exists
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else 'models', exist_ok=True)
        
        # Load weights if available, else use randomly initialized network (useful for verifying pipeline)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained models weights from {model_path}")
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
            
            # Prepare tensor and transfer to device
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # 1. Forward Pass (Inference)
            # Use no_grad for memory efficiency during inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                
            # Define classes: 0 = REAL, 1 = FAKE
            real_prob = probs[0].item()
            fake_prob = probs[1].item()
            
            authenticity_score = real_prob * 100
            prediction = "REAL" if real_prob > fake_prob else "FAKE"
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
                "authenticity_score": round(authenticity_score, 2),
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
                            # Maybe we just want one representative heatmap or all of them
                            heatmaps.append(result["heatmap_base64"])
                    
                current_frame_idx += 1

            cap.release()
            
            if not frame_scores:
                raise ValueError("Could not process any frames from the video.")

            # Aggregate results using simple averaging
            avg_score = sum(frame_scores) / len(frame_scores)
            avg_confidence = sum(frame_confidences) / len(frame_confidences)
            
            # Prediction is FAKE if average score < 50, otherwise REAL
            final_prediction = "REAL" if avg_score >= 50 else "FAKE"
            
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
