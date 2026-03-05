import cv2
import urllib.request
import numpy as np
from ml.inference import DeepfakeDetector

# Download a sample midjourney face image
url = "https://thispersondoesnotexist.com/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    response = urllib.request.urlopen(req)
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
except Exception as e:
    print("Error:", e)
    img = np.zeros((224, 224, 3), dtype=np.uint8)

# Run inference
detector = DeepfakeDetector()
is_success, buffer = cv2.imencode('.jpg', img)
result = detector.predict_image(buffer.tobytes())

print("Test Image Result:", result["prediction"], "Score:", result["authenticity_score"], "Confidence:", result["confidence"])
