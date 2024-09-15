# Retail Demand Forecasting using deep learning

This repository contains a production-ready Flask app, as well as a detailed model training notebook for serving a CNN model trained to predict the sales of a particular SKU offered by a restaurant.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model APIs](#model-apis)
- [Transformation Pipeline](#transformation-pipeline)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [License](#license)

## Project Overview

This project is designed to serve a CNN model trained to forecast the demand of a particular SKU from a restaurant via a Flask API, which can be deployed using a Docker container. A detailed data exploration and model training notebook is available in the 'notebooks' folder. The raw data is available under the 'data' folder. The data was sourced from kaggle. The three files cover 3 years worth of sales data for a fictional restaurant.

In addition to serving the model, the repository includes:
- **Transformation Pipeline:** To preprocess the data for models.
- **Prediction Pipeline:** To make predictions and adjust them based on seasonal and trend adjustments


## Folder Structure

```bash
project-root/
│
├── app/                      # Flask app
│   ├── __init__.py            # Initializes the app
│   ├── app.py                 # Flask API for serving models
│   ├── predictor.py           # Prediction scripts
│   ├── transformation.py      # Data preprocessing steps
│
├── data/                      # Data storage 
│   ├── items.csv                  # Raw data files
│   ├── restaurants.csv           
│   ├── sales_train.csv            
│
├── models/                    # Model training scripts
│   ├── forecastingmodel.h5         # Trained model
│
├── scripts/                   # Helper scripts for running app & pipelines
│   ├── run_app.sh             # Start the Flask app
│
├── notebooks/                 # Jupyter notebooks for EDA & experiments
│   └── Model_trainer.ipynb
│
├── Dockerfile                 # Dockerfile for production deployments
├── requirements.txt           # Python package requirements
├── README.md                  # Project documentation
└── wsgi.py                    # WSGI entry point for production deployment

## Features

- **Model Serving:** Serve a trained CNN model via a Flask API.
- **Data Transformation:** Preprocess data to prepare it for model inference.
- **Modular Architecture:** Clear separation of concerns for transformation, and model serving.
- **Scalable:** Ready for deployment in production environments (WSGI, Docker, etc.).

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/ShubhanKamat/DemandForecasting.git
    cd DemandForecasting
    ```

2. **Set Up Python Virtual Environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```


## Usage

1. **Running the Flask App**

    To start the Flask app locally, run:

    ```bash
    bash scripts/run_app.sh
    ```

    Alternatively, you can run:

    ```bash
    export FLASK_APP=app/app.py
    export FLASK_ENV=production  # Set environment for production
    flask run
    ```

    By default, the Flask API will be served at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Model APIs

The Flask app exposes a route for interacting with the model. The model is preloaded into memory when the app starts.

**Endpoint:**

- `POST /predict`: Predicts using the CNN model.


**Example Request:**

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
    "data": [your_time_series_data]
}'

