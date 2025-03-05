RO-Based Desalination Cost Prediction

Overview

This project uses a multi-layer feedforward neural network with backpropagation to predict the capital cost of Reverse Osmosis (RO) desalination plants. The model is trained on a dataset of 1806 RO plants with capacities of at least 1000 m³/day, utilizing six input parameters.

Features

Capital Cost Prediction: Estimates the investment cost of new RO desalination plants.

Carbon Footprint Calculator: Assesses the environmental impact of desalination plants based on energy consumption and emissions.

Water Quality Estimation: Uses TDS (Total Dissolved Solids) and other parameters to determine water quality and assess its usability.

Dynamic UI: Interactive frontend showcasing all features with seamless navigation.

Dataset

The model is trained on a dataset containing:

Plant Location (categorical)

Plant Capacity (numerical)

Project Award Year (numerical)

Raw Water Salinity (numerical)

Plant Type (categorical)

Project Financing Type (categorical)

Capital Cost (output variable)

Carbon Footprint Data (additional environmental impact metrics)

Water Quality Data (TDS, contaminants, and usability indicators)

Technology Stack

Backend

Machine Learning: PyTorch (Feedforward Neural Network, Backpropagation, Dropout, L2 Regularization, Early Stopping)

Dataset Handling: Pandas, NumPy

Model Training & Evaluation: Scikit-learn, Matplotlib

Frontend

UI Framework: HTML, CSS, JavaScript
