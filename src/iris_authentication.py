
# Iris Authentication Module using Structural Similarity and Hamming Distance

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class IrisAuthenticator:
    def __init__(self):
        pass

    def compute_ssim(self, image1, image2):
        """
        Compute the Structural Similarity Index (SSIM) between two images to measure similarity.
        
        Args:
            image1 (np.array): First input image (grayscale).
            image2 (np.array): Second input image (grayscale).
        
        Returns:
            float: SSIM score between the two images.
            np.array: SSIM image showing the similarity map.
        """
        score, ssim_image = ssim(image1, image2, full=True)
        return score, (ssim_image * 255).astype(np.uint8)

    def highlight_anomalies(self, image1, image2, threshold_value=50):
        """
        Highlight anomalies between two images by calculating the absolute difference and applying thresholding.
        
        Args:
            image1 (np.array): First input image (grayscale).
            image2 (np.array): Second input image (grayscale).
            threshold_value (int): Threshold value for anomaly detection.
        
        Returns:
            np.array: Binary anomaly map highlighting regions of significant differences.
        """
        difference_image = cv2.absdiff(image1, image2)
        _, anomaly_map = cv2.threshold(difference_image, threshold_value, 255, cv2.THRESH_BINARY)
        return anomaly_map

    def compute_hamming_distance(self, iris_code1, iris_code2):
        """
        Compute the Hamming Distance between two binary iris codes to measure dissimilarity.
        
        Args:
            iris_code1 (np.array): Binary iris code for the first image.
            iris_code2 (np.array): Binary iris code for the second image.
        
        Returns:
            float: Hamming distance between the two iris codes.
        """
        return np.sum(np.bitwise_xor(iris_code1, iris_code2)) / len(iris_code1)

    def extract_iris_code(self, image):
        """
        Extract the binary iris code from the given image using Gabor filtering and thresholding.
        
        Args:
            image (np.array): Input iris image (grayscale).
        
        Returns:
            np.array: Binary iris code.
        """
        # Placeholder for actual Gabor filtering and encoding logic
        _, binary_code = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        return binary_code

    def authenticate(self, image1, image2):
        """
        Authenticate the iris patterns by comparing the extracted iris codes from two images.
        
        Args:
            image1 (np.array): First input image (grayscale).
            image2 (np.array): Second input image (grayscale).
        
        Returns:
            float: Hamming distance between the iris codes of the two images.
            bool: True if the Hamming distance is below the threshold, indicating a match.
        """
        code1 = self.extract_iris_code(image1)
        code2 = self.extract_iris_code(image2)
        hamming_distance = self.compute_hamming_distance(code1, code2)
        return hamming_distance, hamming_distance < 0.32  # Typical threshold for iris authentication

    def visualize_comparison(self, image1, image2, ssim_image, anomaly_map):
        """
        Visualize the comparison between two images, showing SSIM map and detected anomalies.
        
        Args:
            image1 (np.array): First input image (grayscale).
            image2 (np.array): Second input image (grayscale).
            ssim_image (np.array): SSIM image showing similarity map.
            anomaly_map (np.array): Anomaly map highlighting differences.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        
        # Show the original images
        plt.subplot(1, 4, 1)
        plt.imshow(image1, cmap='gray')
        plt.title('Image 1')
        
        plt.subplot(1, 4, 2)
        plt.imshow(image2, cmap='gray')
        plt.title('Image 2')
        
        # Show SSIM similarity map
        plt.subplot(1, 4, 3)
        plt.imshow(ssim_image, cmap='gray')
        plt.title('SSIM Map')
        
        # Show anomaly map
        plt.subplot(1, 4, 4)
        plt.imshow(anomaly_map, cmap='gray')
        plt.title('Anomaly Map')
        
        plt.show()
