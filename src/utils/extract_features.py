import cv2
from skimage.feature import hog

def extract_features(segmented_image):
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))  # Resize gambar ke ukuran tetap

    # Ekstraksi fitur HOG
    hog_features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys',
                       visualize=False)
    return (hog_features)
