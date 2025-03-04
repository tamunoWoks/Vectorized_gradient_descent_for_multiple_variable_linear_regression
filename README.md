## Vectorized Gradient Descent for Multiple Variable Linear Regression
This repository contains a Jupyter notebook that demonstrates the implementation of gradient descent for multiple variable linear regression using vectorized operations in Python. The notebook provides a step-by-step guide to understanding and implementing gradient descent, a fundamental optimization algorithm in machine learning.
### Overview
This notebook focuses on solving a linear regression problem with multiple features using gradient descent. It demonstrates how to:
- Implement vectorized operations for efficient computation.
- Compute the cost function (mean squared error).
- Calculate gradients for model parameters (weights and bias).
- Perform gradient descent to optimize the model.
- Visualize the convergence of the algorithm.  

The notebook is designed for educational purposes, and is suitable for beginners, and anyone interested in understanding the fundamentals of gradient descent and linear regression.
### Notebook Structure
The notebook is organized into the following sections:
- **Libraries:** Imports necessary Python libraries such as numpy, matplotlib, and custom styles for plotting.
- **Data Preparation:** Defines the training data (X_train and y_train) for a linear regression problem with multiple features.
- **Model Initialization:** Initializes the model parameters (weights w and bias b).
- **Prediction Functions:**
    - Implements an element-wise prediction function (predict_single_loop).
    - Implements a vectorized prediction function (predict) for efficient computation.
- **Cost Function:** Computes the cost (mean squared error) for the linear regression model.
- **Gradient Calculation:** Calculates the gradients for the weights and bias using vectorized operations.
- **Gradient Descent:** Performs gradient descent to optimize the model parameters and minimize the cost function.
- **Visualization:** Plots the cost versus iteration to monitor the convergence of the algorithm.
- **Observations:** Discusses the results and potential improvements.
### Key Concepts
- Linear Regression
- Gradient Descent
- Vectorization
- Cost/Loss Function
