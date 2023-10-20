# Convolutional Neural Network (CNN) for MNIST Digit Classification
This repository contains a simple Convolutional Neural Network (CNN) model implemented in Python using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The model is designed to recognize and distinguish between the ten different digits (0-9).

## Getting Started
To use and run this code, follow these steps:

### Prerequisites
Make sure you have Python 3.x installed on your system.

Install the required libraries using pip:

`pip install tensorflow`

### Running the Code

1- Clone this repository to your local machine.

`git clone <repository-url>`

2- Open a terminal or command prompt and navigate to the directory where you've cloned this repository.

3- Run the script using the following command:

`python mnist_cnn.py`

This will train the CNN model using the MNIST dataset and display the training progress, including loss and accuracy.


## Code Description

### Importing Required Libraries
The code begins by importing the necessary libraries for working with TensorFlow and Keras.

### Loading Data
The MNIST dataset is loaded using mnist.load_data() method, and the dataset is split into training and testing sets.

### Data Preprocessing
The data is reshaped to have a single channel (grayscale) and normalized by scaling pixel values to the range [0, 1].

### Model Architecture
The CNN model is defined using a Sequential API. The architecture consists of the following layers:

- Convolutional layer with 32 filters and a ReLU activation function.
- Max-pooling layer with a 2x2 pool size.
- Flatten layer to convert the 2D feature maps to a 1D vector.
- Fully connected dense layer with 100 units and a ReLU activation function.
- Output layer with 10 units and a softmax activation function for class probabilities.

### Model Compilation
The model is compiled using the sparse categorical cross-entropy loss, the Adam optimizer, and accuracy as the evaluation metric.

### Model Training
The model is trained on the training data with a specified number of epochs.

### Contributing
Feel free to contribute to this project by making improvements, fixing bugs, or adding new features. Create a pull request with your changes.

## Acknowledgments
- This code is a simple example of using CNNs for image classification.
- The MNIST dataset is widely used for digit recognition tasks in the machine learning community.
