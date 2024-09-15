import os
import pandas as pd
import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

# Create 'logs' directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
logging.basicConfig(
    filename='logs/transformation.log',  # Log file for this module
    level=logging.DEBUG,  # Set logging level to capture all events (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

# Function to preprocess input data, decompose seasonal effects, and create additional time-based features
def transform_input(data):
    """
    Preprocess the input data by parsing dates, calculating daily averages, 
    detrending the time series, and adding day-of-week and day-of-year features.

    Parameters:
    - data: Input raw data (assumed to have 'date' and 'item_count' columns)

    Returns:
    - df_plot: Preprocessed data with date, item_count, day_of_week, and day_of_year columns
    - adjustment: Seasonal decomposition result for the item_count (useful for trend analysis)
    """
    try:
        # Convert input to DataFrame (if not already)
        logging.info("Starting input data transformation.")
        df = pd.DataFrame(data)

        # Ensure 'date' column is in datetime format (handle invalid parsing using 'coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        logging.debug(f"Parsed 'date' column with shape: {df['date'].shape}")

        # Group by 'date' and compute mean of 'item_count' for each date (aggregating data)
        df_plot = df[['date', 'item_count']].groupby(['date']).mean().reset_index()
        logging.info("Grouped data by 'date' and calculated daily average.")

        # Decompose the item_count time series into seasonal, trend, and residual components
        adjustment = seasonal_decompose(df_plot.item_count, model='multiplicative')
        logging.info("Performed seasonal decomposition on item_count.")

        # Remove the trend component from the item_count using detrend (linear trend removal)
        df_plot.item_count = signal.detrend(df_plot.item_count)
        logging.info("Detrended the item_count series.")

        # Add day-of-week (0=Monday, 6=Sunday) and day-of-year (1-365) features for the date
        df_plot['dow'] = df_plot['date'].dt.dayofweek
        df_plot['doy'] = df_plot['date'].dt.dayofyear
        logging.debug(f"Added day-of-week and day-of-year features. Shape: {df_plot.shape}")

        return df_plot, adjustment

    except Exception as e:
        logging.error(f"Error during data transformation: {e}")
        raise

# Function to convert a time series into a supervised learning problem (shifting data for past and future steps)
def series_to_supervised(data, window=1, lag=1, dropnan=True):
    """
    Converts a time series dataset into a supervised learning format, creating lagged values for prediction.
    
    Parameters:
    - data: DataFrame containing the time series data.
    - window: Number of past time steps (lags) to include.
    - lag: Number of future steps to forecast.
    - dropnan: If True, drops rows with NaN values (resulting from shifting).

    Returns:
    - agg: DataFrame containing lagged and future values (with t-steps and t+steps).
    """
    try:
        logging.info(f"Converting time series data to supervised format with window={window} and lag={lag}.")
        cols, names = list(), list()

        # Create lagged features for past time steps (t-window to t-1)
        for i in range(window, 0, -1):
            cols.append(data.shift(i))
            names += [('%s(t-%d)' % (col, i)) for col in data.columns]
        logging.debug(f"Created lagged features for past {window} time steps.")

        # Current time step (t)
        cols.append(data)
        names += [('%s(t)' % (col)) for col in data.columns]

        # Create future step (t+lag) for supervised learning (target to predict)
        cols.append(data.shift(-lag))
        names += [('%s(t+%d)' % (col, lag)) for col in data.columns]

        # Concatenate all the columns (lags, current, future) horizontally
        agg = pd.concat(cols, axis=1)
        agg.columns = names

        # Drop rows with NaN values (from shifting) if required
        if dropnan:
            agg.dropna(inplace=True)
        logging.info(f"Generated supervised dataset with shape: {agg.shape}")

        return agg

    except Exception as e:
        logging.error(f"Error in converting series to supervised format: {e}")
        raise

# Function to prepare input features for the model using past time steps (window) and future prediction span
def prepare_model_input(data, window=29, future_span=30):
    """
    Prepares the input data for model training by generating supervised learning features (lags and future values).
    
    Parameters:
    - data: DataFrame containing the time series data with item, store, and other features.
    - window: The number of past time steps to consider (history).
    - future_span: The number of future time steps to forecast.

    Returns:
    - x: A 3D array with concatenated features for model input.
    """
    try:
        logging.info(f"Preparing model input with window={window} and future_span={future_span}.")

        # Convert the time series data into supervised learning format (with window and future_span)
        series = series_to_supervised(data, window=window, lag=future_span)

        # Feature consistency check: Ensure store and item remain constant over the window
        last_item = 'item(t-%d)' % window
        last_store = 'store(t-%d)' % window
        series = series[(series['store(t)'] == series[last_store])]
        series = series[(series['item(t)'] == series[last_item])]

        # Extract the sales, day-of-week, and day-of-year series from the supervised dataset
        sales_series = series['adjust'].values
        dow_series = series['dow'].values
        doy_series = series['doy'].values

        # Reshape each series to match the expected input shape for model training
        t1 = sales_series.reshape(sales_series.shape + (1,))
        t2 = dow_series.reshape(dow_series.shape + (1,))
        t3 = doy_series.reshape(doy_series.shape + (1,))

        # Concatenate the features along the last axis to form the final input array for the model
        x = np.concatenate([t1, t2, t3], axis=2)
        logging.debug(f"Prepared model input data with shape: {x.shape}")

        return x

    except Exception as e:
        logging.error(f"Error in preparing model input: {e}")
        raise
