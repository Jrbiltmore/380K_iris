
# Unit Tests for Iris Detection Functions

import unittest
import cv2
import numpy as np
from src.iris_detection import IrisDetector

class TestIrisDetection(unittest.TestCase):

    def setUp(self):
        # Initialize the IrisDetector instance
        self.detector = IrisDetector(input_shape=(128, 128, 3), num_classes=2)
        # Create a sample image (RGB)
        self.image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    def test_build_cnn(self):
        # Test CNN model building
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.model.input_shape, (None, 128, 128, 3))
        self.assertEqual(self.detector.model.output_shape, (None, 2))

    def test_predict(self):
        # Test model prediction on a sample image
        predictions = self.detector.predict(self.image)
        self.assertEqual(predictions.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()

    def test_train(self):
        # Mock directories for train and validation data
        train_dir = 'path/to/train_data'
        val_dir = 'path/to/val_data'

        # Ensure that the train method doesn't throw an error (mock training)
        try:
            history = self.detector.train(train_dir, val_dir, batch_size=2, epochs=1)
        except Exception as e:
            self.fail(f"Training raised an exception: {e}")

    def test_evaluate(self):
        # Mock test directory
        test_dir = 'path/to/test_data'

        # Ensure that the evaluate method runs without error (mock evaluation)
        try:
            results = self.detector.evaluate(test_dir, batch_size=2)
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"Evaluation raised an exception: {e}")

    def test_save_load_model(self):
        # Mock file path for model saving
        model_path = 'path/to/saved_model.h5'

        # Test saving the model
        try:
            self.detector.save_model(model_path)
        except Exception as e:
            self.fail(f"Saving model raised an exception: {e}")

        # Test loading the model
        try:
            self.detector.load_model(model_path)
        except Exception as e:
            self.fail(f"Loading model raised an exception: {e}")

    def test_visualize_training(self):
        # Mock history object
        mock_history = {
            'accuracy': [0.8, 0.85],
            'val_accuracy': [0.75, 0.8],
            'loss': [0.4, 0.35],
            'val_loss': [0.45, 0.4]
        }

        # Ensure visualization runs without error
        try:
            self.detector.visualize_training(mock_history)
        except Exception as e:
            self.fail(f"Visualization raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
