from app.app import app
import logging

# Create 'logs' directory if it doesn't exist
import os
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
logging.basicConfig(
    filename='logs/app.log',  # Log file for the Flask application
    level=logging.INFO,  # Set logging level to INFO to capture basic operational messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

if __name__ == "__main__":
    logging.info("Starting Flask application.")
    try:
        app.run(debug=True)  # Set debug=True for development; set to False for production
    except Exception as e:
        logging.error(f"Error occurred while running the Flask application: {e}")
