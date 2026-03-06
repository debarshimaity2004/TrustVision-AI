import cv2
import urllib.request
import numpy as np
from scipy.fftpack import dct

def analyze_image(url, label):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    except Exception as e:
        print(f"Error fetching {label}: {e}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # DCT
    dct_y = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    power_spectrum = np.abs(dct_y) ** 2
    h, w = power_spectrum.shape
    high_freq_energy = np.sum(power_spectrum[int(h*0.5):, int(w*0.5):])
    total_energy = np.sum(power_spectrum)
    hf_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    # Chrominance
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycbcr)
    cr_var = cr.var()
    cb_var = cb.var()
    
    # Edge Density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (img.shape[0]*img.shape[1])

    print(f"--- {label} ---")
    print(f"  Laplacian Var: {laplacian_var:.2f}")
    print(f"  DCT HF Ratio : {hf_ratio:.6f}")
    print(f"  Cr Var       : {cr_var:.2f}")
    print(f"  Cb Var       : {cb_var:.2f}")
    print(f"  Edge Density : {edge_density:.2f}")

# Midjourney AI Face
analyze_image("https://thispersondoesnotexist.com/", "AI FAKE")

# Real Face (Wikipedia)
analyze_image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Woman_with_freckles.jpg/800px-Woman_with_freckles.jpg", "REAL")
