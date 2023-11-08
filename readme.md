# Bike Rental Prediction

This project aims to predict bike rentals based on historical data and weather conditions.

## Table of Contents

- [Bike Rental Prediction](#bike-rental-prediction)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)

## Description

The project uses a machine learning model (LightGBM) to predict bike rentals by considering various features, including historical data and weather conditions. It offers functionality to train, monitor, and update the model.

## Features

- Train the model with data splits
- Get inference on the existing model
- Train the model on all available data and save it
- Update the model by training on new data
- Predict future bike rentals based on weather conditions
- Update LightGBM hyperparameters

## Getting Started

To get started with the project, follow these steps:

### Prerequisites

Before running the project, ensure you have the following dependencies installed:

- Python 3.8+
- Required libraries: NumPy, Pandas, LightGBM


### Installation

Install the required libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

## Usage 

Here are some examples of how to use the project:

   ```bash
   ## Training model with data splits 
   python main.py --task_type 0

   ## Getting inference on the existing model:
   python main.py --task_type 1

   ## Training the model on all available data and saving it:
   python main.py --task_type 2

   ## Updating the model by training on new data:
   python main.py --task_type 3

   ## Predicting future bike rentals:
   python main.py --task_type 4

   ## Updating LightGBM hyperparameters:
   python main.py --task_type 5
   ```