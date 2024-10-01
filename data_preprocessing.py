import cv2
import numpy as np
from skimage import exposure

def apply_clahe(image):
    """Apply CLAHE to enhance contrast of the image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def normalize_image(image):
    """Normalize image to [0,1] range."""
    return image / 255.0

def preprocess_image(image):
    """Preprocess image by applying CLAHE and normalization."""
    clahe_img = apply_clahe(image)
    return normalize_image(clahe_img)
