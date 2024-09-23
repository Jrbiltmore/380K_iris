
# Unit Tests for Iris Authentication Functions

import unittest
import cv2
import numpy as np
from src.iris_authentication import IrisAuthenticator

class TestIrisAuthentication(unittest.TestCase):

    def setUp(self):
        # Initialize the IrisAuthenticator instance
        self.authenticator = IrisAuthenticator()
        # Create two sample images with slight differences
        self.image1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        self.image2 = self.image1.copy()
        self.image2[32:40, 32:40] = 0  # Add a small anomaly in the second image

    def test_compute_ssim(self):
        # Test SSIM computation between two images
        score, ssim_image = self.authenticator.compute_ssim(self.image1, self.image2)
        self.assertIsNotNone(ssim_image)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_highlight_anomalies(self):
        # Test highlighting anomalies between two images
        anomaly_map = self.authenticator.highlight_anomalies(self.image1, self.image2)
        self.assertIsNotNone(anomaly_map)
        self.assertEqual(anomaly_map.shape, self.image1.shape)

if __name__ == '__main__':
    unittest.main()

    def test_extract_iris_code(self):
        # Test iris code extraction from an image
        binary_code = self.authenticator.extract_iris_code(self.image1)
        self.assertIsNotNone(binary_code)
        self.assertEqual(binary_code.shape, self.image1.shape)

    def test_compute_hamming_distance(self):
        # Test computation of Hamming distance between two iris codes
        code1 = self.authenticator.extract_iris_code(self.image1)
        code2 = self.authenticator.extract_iris_code(self.image2)
        hamming_distance = self.authenticator.compute_hamming_distance(code1, code2)
        self.assertGreaterEqual(hamming_distance, 0)
        self.assertLessEqual(hamming_distance, 1)

    def test_authenticate(self):
        # Test iris authentication between two images
        hamming_distance, is_match = self.authenticator.authenticate(self.image1, self.image2)
        self.assertIsInstance(hamming_distance, float)
        self.assertIsInstance(is_match, bool)

    def test_visualize_comparison(self):
        # Test visualization of comparison between two images
        score, ssim_image = self.authenticator.compute_ssim(self.image1, self.image2)
        anomaly_map = self.authenticator.highlight_anomalies(self.image1, self.image2)

        try:
            self.authenticator.visualize_comparison(self.image1, self.image2, ssim_image, anomaly_map)
        except Exception as e:
            self.fail(f"Visualization raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
