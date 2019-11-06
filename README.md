# Introduction
The purpose of this lab is to compare the performance of several different machine learning algorithms in identifying items of clothing via the Fashion MNIST dataset. This dataset is currently used as a benchmark for computer vision problems since modern neural networks and classic machine learning algorithms can achieve high accuracy on the classic MNIST dataset out of the box.

# Methodology
The Fashion-MNIST dataset was obtained from Keras, a high-level API for TensorFlow. It consists of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig1.jpg)

Grayscale pixels go from 0 to 255, these were normalized by dividing by 255 to make all numbers  

## Pre-Processing
There were three main pre-processing steps for the different algorithms:
* Grayscale pixels go from 0 to 255. The dataset was normalized by dividing by 255 to make all the numbers range from 0 to 1.
* Some pre-processing was done for the sklearn classifiers, namely flattening the numpy arrays from the 28 by 28 matrices to single vectors of length 784.
* The convolution neural networks need the input dataset to be 4 dimensional, the reshape command was used to transform the data. For example:
  * Original shape: (60000, 28, 28)
  * New shape: (60000, 28, 28, 1)

The dataset was down sampled when training and testing the classic machine learning algorithms from Sklearn because they were taking an extremely long time to run on the full dataset.
* A random sample of 20,000 examples was used for training
* A random sample of 4,000 examples was used for testing

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

<p float="middle">
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig6.jpg" width="250" />
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig7.jpg" width="250" /> 
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig8.jpg" width="250" />
</p>

The rectified linear unit activation seemed to be the best for this problem, although they all seem to eventually converge regardless. Learning rate however, did have a large impact when increased so it was kept at the default 0.001 for all the architectures.

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig9.jpg)

# Results and Findings
In my experiment, the neural networks are the clear winners. Sorted by accuracy on the testing set (green), the four neural networks come out on top, they also show much less overfitting.  

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig10.jpg)

The neural networks are not the fastest, but their training time is comparable to the rest of the algorithms tested. There are some notable exceptions like the gradient boost classifier, with average accuracy and a massive training time, or naïve Bayes which is extremely fast but inaccurate for this problem.

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig11.jpg)

The hard categories seem to be common for all the different algorithms.

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig12.jpg)

Printing out some of the mislabeled items from my best model (CNN #1) shows how hard it is to label some of these images:

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig13.jpg)

Some notable examples:
<p float="middle">
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig14.jpg" width="200" />
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig15.jpg" width="200" /> 
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig16.jpg" width="200" />
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig17.jpg" width="200" />
</p>

# Discussion and Conclusion
*	According to my experiment with the Fashion MNST dataset, a convolution neural network is the best option for identifying the items of clothing in this dataset
*	My best model was a CNN with 2 convolution layers (both with a rectified linear unit activation) and 2 pooling layers followed by one, large, fully connected layer and a dropout layer.
  * 0.9142 on the Test set
  
<p float="middle">
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig18.jpg" width="350" />
  <img src="https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig19.jpg" width="350" /> 
</p>

*	In my opinion, Deep Learning is ideal for computer vision problems not only because of its accuracy and its ability to model extremely complex decision boundaries, but because the features (individual pixels in an image) in these problems do not do that much for interpretation regardless.
*	The generalization power in the neural networks and the ability to save and initialize a model from a file through a high level API like Keras, more than makes up for the slower compute time.
*	Performance summary table

![Alt Text](https://github.com/pmb06d/computer_vision_fashionMNIST/blob/master/graphs/fig20.jpg)

