import os
import logging
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from ingestion import fetch_all_data  # Custom module for data ingestion
from transformation import transform_input, prepare_model_input  # Custom modules for data transformation
from prediction import load_model, predict_sales  # Custom modules for loading model and making predictions

# Create 'logs' directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
logging.basicConfig(
    filename='logs/app.log',  # Log file
    level=logging.DEBUG,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

# Initialize Flask application
app = Flask(__name__)

# Log startup info
logging.info("Starting Flask application and loading the model...")

try:
    # Load the pre-trained model once when the app starts to avoid reloading it on every request
    model = load_model('models/forecastingmodel.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    """
    Flask route to handle POST requests for sales predictions.

    Expects input data in JSON format, transforms it, prepares it for the model,
    and returns the predicted sales in JSON format.

    Returns:
    - JSON response containing the sales prediction.
    """
    try:
        # Step 1: Retrieve input data from the incoming POST request (in JSON format)
        logging.info("Received a new prediction request.")
        data = request.json

        if not data:
            logging.warning("No data provided in the request.")
            return jsonify({'error': 'No data provided'}), 400

        # Step 2: Transform the raw input data to extract meaningful features (including seasonal adjustments)
        logging.debug("Transforming input data...")
        transformed_data, adjustment = transform_input(data)
        logging.debug("Input data transformed successfully.")

        # Step 3: Prepare the transformed data for model input (using past time steps and future forecasting span)
        logging.debug("Preparing model input data...")
        model_input = prepare_model_input(transformed_data)
        logging.debug("Model input prepared successfully.")

        # Step 4: Use the pre-trained model to make a sales prediction
        logging.debug("Making prediction using the model...")
        prediction = predict_sales(model, model_input, adjustment)
        logging.info("Prediction made successfully.")

        # Step 5: Return the prediction as a JSON response (convert NumPy arrays to lists for JSON serialization)
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        # Log any exception that occurs during the prediction process
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Entry point for running the Flask application
if __name__ == '__main__':
    # Enable debug mode for development (shows errors and reloads on changes)
    logging.info("Starting the Flask app in debug mode.")
    app.run(debug=True)
