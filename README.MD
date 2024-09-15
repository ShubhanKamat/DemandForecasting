# ML Model Serving with Flask

This repository contains a production-ready Flask app for serving multiple machine learning models (ES, ARIMA, Prophet, LSTM). It also includes pipelines for data ingestion, transformation, and automated retraining of models.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model APIs](#model-apis)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Transformation Pipeline](#transformation-pipeline)
- [Retraining Models](#retraining-models)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview

This project is designed to serve time-series forecasting models via a Flask API. The app allows users to interact with several models (Exponential Smoothing, ARIMA, Prophet, LSTM) to predict future data trends.

In addition to serving models, the repository includes:
- **Ingestion Pipeline:** To fetch and load data from external sources (e.g., CSV, database).
- **Transformation Pipeline:** To preprocess the data for models.
- **Retraining Pipeline:** For automatically retraining models with new data.

## Folder Structure

```bash
project-root/
│
├── app/                      # Flask app
│   ├── __init__.py            # Initializes the app
│   ├── models/                # Trained models
│   ├── api.py                 # Flask API for serving models
│   ├── ingestion.py           # Data ingestion logic
│   ├── transformation.py      # Data preprocessing steps
│   ├── retrain.py             # Model retraining scripts
│   ├── utils.py               # Helper functions for the app
│   └── config.py              # App configuration
│
├── data/                      # Data storage (raw & processed)
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data files
│
├── models/                    # Model training scripts
│   ├── arima_model.py         # ARIMA model training
│   ├── es_model.py            # Exponential Smoothing training
│   ├── prophet_model.py       # Prophet model training
│   ├── lstm_model.py          # LSTM model training
│
├── scripts/                   # Helper scripts for running app & pipelines
│   ├── run_app.sh             # Start the Flask app
│   ├── retrain_models.sh      # Run retraining pipeline
│
├── notebooks/                 # Jupyter notebooks for EDA & experiments
│   └── eda-and-model-building-es-arima-prophet-lstm.ipynb
│
├── requirements.txt           # Python package requirements
├── README.md                  # Project documentation
└── wsgi.py                    # WSGI entry point for production deployment
