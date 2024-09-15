import os
import numpy as np
import logging
from tensorflow.keras.models import load_model as keras_load_model

# Create 'logs' directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
logging.basicConfig(
    filename='logs/predictor.log',  # Log file for this module
    level=logging.DEBUG,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

# Function to load a pre-trained model from a specified file path
def load_model(model_path):
    """
    Loads a pre-trained Keras model from the specified file path.

    Parameters:
    - model_path: The file path where the Keras model is stored.

    Returns:
    - model: The loaded Keras model.
    """
    try:
        logging.info(f"Attempting to load model from path: {model_path}")
        model = keras_load_model(model_path)
        logging.info(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise

# Function to make sales predictions and adjust them based on seasonal and trend adjustments
def predict_sales(model, input_data, adjustment):
    """
    Predict future sales using the trained model and apply adjustments based on seasonal and trend factors.

    Parameters:
    - model: The trained Keras model used for making predictions.
    - input_data: 3D NumPy array of input features for prediction (shape: [samples, timesteps, features]).
    - adjustment: A seasonal_decompose object containing the seasonal and trend components for adjustment.

    Returns:
    - adjusted_prediction: A NumPy array containing the adjusted sales predictions.
    """
    try:
        # Step 1: Make raw predictions using the trained model
        logging.info("Making prediction using the model.")
        prediction = model.predict(input_data)  # Prediction will return a 2D array of shape (samples, 1)
        logging.debug(f"Raw prediction shape: {prediction.shape}")

        # Step 2: Flatten the predictions to make it a 1D array for easier manipulation
        prediction = prediction.flatten()
        logging.debug(f"Flattened prediction: {prediction}")

        # Step 3: Adjust the predictions based on seasonal and trend factors from the seasonal_decompose object
        logging.info("Adjusting prediction based on seasonal and trend factors.")
        adjusted_prediction = prediction * adjustment.seasonal.iloc[-1] * adjustment.trend.iloc[-1]
        logging.debug(f"Adjusted prediction: {adjusted_prediction}")

        return adjusted_prediction

    except Exception as e:
        logging.error(f"Error during prediction and adjustment: {e}")
        raise
