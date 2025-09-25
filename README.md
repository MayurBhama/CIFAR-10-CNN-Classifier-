# CIFAR-10 CNN: From Overfitting to Optimization

This project provides an end-to-end demonstration of building, diagnosing, and optimizing a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset.

The core focus is to showcase the iterative process of machine learning development, moving from a simple, overfitting baseline to a robust, regularized model.

## The Experimental Process

The project follows a clear, two-stage experiment:
1.  **Baseline Model**: A simple CNN is constructed. While it learns the training data, its performance on validation data is poor, revealing significant overfitting.
2.  **Optimized Model**: To address overfitting, a second model is built incorporating several key regularization and optimization techniques:
    - **Data Augmentation**: Artificially expands the training dataset by applying random transformations to images.
    - **Batch Normalization**: Stabilizes and accelerates the training process.
    - **Dropout**: Prevents feature co-dependency by randomly deactivating neurons during training.
    - **Learning Rate Scheduling**: Dynamically adjusts the learning rate to allow for more precise convergence via a Keras callback.
    - **Early Stopping**: Halts training automatically when validation performance stops improving, saving time and preventing further overfitting.

## Results: A Visual Comparison

The effectiveness of the optimization techniques is clearly visible in the training history.

### Final Test Accuracy
| Model                       | Test Accuracy |
| --------------------------- | :-----------: |
| Baseline Model (Overfitting)| **71.66%*        |
| **Optimized Model**         | **80.53%**   |

### Baseline Model Performance (Severe Overfitting)
*The large gap between the training and validation curves is a classic sign of overfitting.*
<img width="1001" height="393" alt="download (1)" src="https://github.com/user-attachments/assets/72e04180-3b8b-4c17-a43e-f06525a02702" />


### Optimized Model Performance (Regularization Success)
*The training and validation curves now track each other closely, indicating that the model is generalizing well.*
<img width="1001" height="470" alt="download" src="https://github.com/user-attachments/assets/2c7aaea4-5dda-40b7-bb47-3bd819ffc07c" />


## Project Structure

A modular structure is used to separate concerns, making the code clean and scalable.

