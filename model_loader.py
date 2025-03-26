import os
import tensorflow as tf

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        try:
            # Check file extension
            if self.model_path.endswith('.keras'):
                print(f"Loading Keras model from: {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
            elif os.path.isdir(self.model_path):
                print(f"Loading SavedModel from directory: {self.model_path}")
                self.model = tf.saved_model.load(self.model_path)
            else:
                raise ValueError("Unsupported model format. Must be .keras file or SavedModel directory")

            return self.model

        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict(self, input_data):
        """
        Makes predictions using the loaded model.
        Handles differences in how predictions are made between the two formats.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Handle SavedModel format
            if isinstance(self.model, tf.saved_model.SavedModel):
                results = self.model.signatures['serving_default'](input_data)
                return results['classifier']  # Adjust output key as needed

            # Handle Keras model format
            else:
                return self.model.predict(input_data)

        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
