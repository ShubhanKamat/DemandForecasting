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
