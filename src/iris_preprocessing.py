
# Iris Image Preprocessing Module

import cv2
import numpy as np
from skimage import exposure

class IrisPreprocessor:
    def __init__(self):
        pass

    def normalize_image(self, image):
        """
        Normalize image pixel intensities to the range 0-255.
        
        Args:
            image (np.array): Input image.
        
        Returns:
            np.array: Normalized image.
        """
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    def resize_image(self, image, target_size=(128, 128)):
        """
        Resize the input image to a standard size for further processing.
        
        Args:
            image (np.array): Input image.
            target_size (tuple): Target size as (width, height).
        
        Returns:
            np.array: Resized image.
        """
        return cv2.resize(image, target_size)

    def apply_clahe(self, image):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.
        
        Args:
            image (np.array): Input grayscale image.
        
        Returns:
            np.array: Image with enhanced contrast.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def sharpen_image(self, image):
        """
        Sharpen the input image using a kernel.
        
        Args:
            image (np.array): Input image.
        
        Returns:
            np.array: Sharpened image.
        """
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def apply_gaussian_blur(self, image, kernel_size=(5, 5)):
        """
        Apply Gaussian blur to the input image.
        
        Args:
            image (np.array): Input image.
            kernel_size (tuple): Size of the Gaussian kernel.
        
        Returns:
            np.array: Blurred image.
        """
        return cv2.GaussianBlur(image, kernel_size, 0)

    def equalize_histogram(self, image):
        """
        Apply histogram equalization to improve image contrast.
        
        Args:
            image (np.array): Input grayscale image.
        
        Returns:
            np.array: Image with equalized histogram.
        """
        return cv2.equalizeHist(image)

    def adaptive_thresholding(self, image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
        """
        Apply adaptive thresholding to the input image to binarize the image based on local pixel intensity.
        
        Args:
            image (np.array): Input grayscale image.
            max_value (int): Maximum pixel intensity after thresholding.
            adaptive_method (cv2 method): Adaptive thresholding method (e.g., Gaussian or Mean).
            threshold_type (cv2 method): Threshold type (e.g., binary).
            block_size (int): Size of a pixel neighborhood used to calculate the threshold value.
            C (int): Constant subtracted from the mean or weighted mean.
        
        Returns:
            np.array: Binarized image after adaptive thresholding.
        """
        return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)
    
    def rotate_image(self, image, angle):
        """
        Rotate the input image by the specified angle.
        
        Args:
            image (np.array): Input image.
            angle (float): Angle by which the image should be rotated.
        
        Returns:
            np.array: Rotated image.
        """
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
