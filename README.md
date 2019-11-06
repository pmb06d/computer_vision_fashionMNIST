# Introduction
The purpose of this lab is to compare the performance of several different machine learning algorithms in identifying items of clothing via the Fashion MNIST dataset. This dataset is currently used as a benchmark for computer vision problems since modern neural networks and classic machine learning algorithms can achieve high accuracy on the classic MNIST dataset out of the box.

# Methodology
The Fashion-MNIST dataset was obtained from Keras, a high-level API for TensorFlow. It consists of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig1.jpg)

Grayscale pixels go from 0 to 255, these were normalized by dividing by 255 to make all numbers  

## Pre-Processing
There were three main pre-processing steps for the different algorithms:
*	Grayscale pixels go from 0 to 255. The dataset was normalized by dividing by 255 to make all the numbers range from 0 to 1.
*	Some pre-processing was done for the sklearn classifiers, namely flattening the numpy arrays from the 28 by 28 matrices to single vectors of length 784.
*	The convolution neural networks need the input dataset to be 4 dimensional, the reshape command was used to transform the data. For example:
  *	Original shape: (60000, 28, 28)
  *	New shape: (60000, 28, 28, 1)

The dataset was down sampled when training and testing the classic machine learning algorithms from Sklearn because they were taking an extremely long time to run on the full dataset.
*	A random sample of 20,000 examples was used for training
*	A random sample of 4,000 examples was used for testing

## Algorithm Selection
There were a total of 6 classic ML algorithms from Sklearn tested: 
Logistic Regression, k nearest neighbors, Gaussian Naïve Bayes, random forests, gradient boost classifier and support vector machines.

There were also 4 neural networks built suing Keras. These used the following algorithms / architectures:
*	MLP #1: A neural net with 2 fully connected layers. The first one uses a rectified linear unit activation function, the second one uses a sigmoid activation function and the output layer uses softmax to output the probability for each category. The Adam optimizer worked the best with this particular architecture

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig2.jpg)

*	MLP #2: A neural net with 2 fully connected layers and 2 drop out layers. Activation functions are the same as the first model

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig3.jpg)

*	CNN #1: 2 convolution layers (both with a rectified linear unit activation) and 2 pooling layers followed by one, large, fully connected layer and a dropout layer.

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig4.jpg)

*	CNN #2: A simpler CNN with only one convolution layer, this architecture also includes a dropout layer after the pooling layer and a smaller (when compared to the first CNN) fully connected layer before the output layer.

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig5.jpg)

## Deep Learning Model Training
A large (150) number of epochs was used originally to see what the impact would be before settling on a smaller number (25) for the final models, it seems the accuracy on the validation set plateaus fast

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig6.jpg) ![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig7.jpg) ![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig8.jpg)
