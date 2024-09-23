
# Iris Detection Module using CNN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class IrisDetector:
    def __init__(self, input_shape=(128, 128, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_cnn()

    def build_cnn(self):
        """
        Build a CNN model for iris detection.
        
        Returns:
            model: Compiled CNN model.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_dir, val_dir, batch_size=32, epochs=50):
        """
        Train the CNN model for iris detection.
        
        Args:
            train_dir (str): Directory with training images.
            val_dir (str): Directory with validation images.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for training.
        
        Returns:
            history: Training history object.
        """
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
                                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                           horizontal_flip=True, fill_mode="nearest")

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(train_dir, target_size=self.input_shape[:2],
                                                            batch_size=batch_size, class_mode='categorical')

        val_generator = val_datagen.flow_from_directory(val_dir, target_size=self.input_shape[:2],
                                                        batch_size=batch_size, class_mode='categorical')

        history = self.model.fit(train_generator, epochs=epochs, validation_data=val_generator)
        return history

    def evaluate(self, test_dir, batch_size=32):
        """
        Evaluate the CNN model for iris detection on test data.
        
        Args:
            test_dir (str): Directory with test images.
            batch_size (int): Batch size for evaluation.
        
        Returns:
            dict: Evaluation results (loss and accuracy).
        """
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(test_dir, target_size=self.input_shape[:2],
                                                          batch_size=batch_size, class_mode='categorical')
        results = self.model.evaluate(test_generator)
        return dict(zip(self.model.metrics_names, results))

    def save_model(self, model_path):
        """
        Save the trained CNN model to the specified path.
        
        Args:
            model_path (str): Path to save the model.
        """
        self.model.save(model_path)

    def load_model(self, model_path):
        """
        Load a pre-trained CNN model from the specified path.
        
        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image):
        """
        Predict whether the given image contains modulated iris patterns.
        
        Args:
            image (np.array): Input image (preprocessed).
        
        Returns:
            np.array: Predicted class probabilities.
        """
        image = image / 255.0
        image = image.reshape(1, *self.input_shape)
        return self.model.predict(image)

    def visualize_training(self, history):
        """
        Visualize the training performance of the CNN model (accuracy and loss over epochs).
        
        Args:
            history: Training history object returned by the model fit method.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
