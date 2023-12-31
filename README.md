# Digit Recognizer

A deep learning project for recognizing English digits using Convolutional Neural Networks (CNN).

## Project Description

The Digit Recognizer project is designed to recognize handwritten English digits using a Convolutional Neural Network (CNN). This project is implemented in Python using TensorFlow and provides a user-friendly interface using Tkinter for drawing digits, predicting the handwritten digit, and visualizing the model's training history.

## Purpose

The purpose of this project is to showcase the application of deep learning techniques in image recognition, specifically for recognizing handwritten digits.

## Usage

There is a pre-trained file ```digit_recognition_cnn_model.keras``` that has been given in this repository.

To use the Digit Recognizer, follow these steps:

1. Run the main script, `gui.py`.
2. Draw a digit on the canvas provided in the GUI.
3. Click the "Predict" button to see the model's prediction.
4. Use the "Clear Canvas" button to reset the canvas for a new input.

## Model Training 
The CNN model used in this project is trained on the MNIST dataset.

Training Details
The model is built using TensorFlow's Keras API. It consists of convolutional layers, max-pooling layers, and fully connected layers.

## Results

### Model Accuracy

The trained CNN model achieved an accuracy of 98.8% on the validation set.

### Prerequisites

Make sure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

## Visualizations

### Initial Data

![initial_data_plot](https://github.com/paramsureliya/digit-recognizer/assets/148708744/6db15675-a65c-4cfe-9120-92e8640805bb)

### Training and Validation Metrics

![loss_plot](https://github.com/paramsureliya/digit-recognizer/assets/148708744/0dbcf867-06dc-4625-99de-f293f2f3a9d2)


### Sample Predictions

![sample_prediction](https://github.com/paramsureliya/digit-recognizer/assets/148708744/2afbde64-070b-4de2-af68-b45208e3e143)
