import cv2
import urllib.request
import numpy as np

url = "https://thispersondoesnotexist.com/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

print(f"Feature - Laplacian Variance: {laplacian_var}")

# Get edge density
edges = cv2.Canny(gray, 100, 200)
edge_density = np.sum(edges) / (img.shape[0]*img.shape[1])
print(f"Feature - Edge Density: {edge_density}")
