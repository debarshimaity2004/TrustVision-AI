import cv2
import urllib.request
import numpy as np

# Load local testing image
img = cv2.imread("C:/Users/Taaru/OneDrive/Desktop/NLP project/test.jpg") # I'll just write code to test this specific face structure

# To simulate testing this directly:
def analyze_direct_image(img_path):
    import base64
    from ml.inference import DeepfakeDetector
    
    # Run the raw heuristics printout manually 
    img = cv2.imread(img_path)
    if img is None:
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    from scipy.fftpack import dct
    dct_y = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    power_spectrum = np.abs(dct_y) ** 2
    h, w = power_spectrum.shape
    hf_ratio = np.sum(power_spectrum[int(h*0.5):, int(w*0.5):]) / np.sum(power_spectrum)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0]*gray.shape[1])
    
    print("\n--- Raw Values for Real Girl ---")
    print("Laplacian (Smoothness):", laplacian_var)
    print("DCT HF Ratio:", hf_ratio)
    print("Edge Density:", edge_density)
    
analyze_direct_image("C:/Users/Taaru/OneDrive/Desktop/NLP project/TrustVision-AI/test.jpg")
