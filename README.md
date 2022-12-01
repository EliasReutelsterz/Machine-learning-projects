# Machine-learning-projects
University projects in the courses COSE474 Deep Learning and MATH292 Mathematics for AI from Korea University.
In every program the dataset still has to be loaded. Information is provided in the corresponding exercise sheet.

## Deep Learning Project 1 (MLP Implementation):
This project contains a two layered neural network from scratch. It can calculate a forecast for the input data (in this case CIFAR-10 dataset) and is able to be trained via backpropagation. The specific exercises and the model infrastructure can be looked up in the pdf-files. I found the original project at the Stanford website <a href="https://www.w3schools.com](http://cs231n.stanford.edu/">CS231n: Convolutional Neural Networks for Visual Recognition</a>.

## Deep Learning Project 2 (CNN Architecture Implementation):
This project treats a pytorch environment in which the CIFAR10 dataset should be used. First of all the so called VGG-16 model was used and afterwards we had to fill in the missing pieces of the ResNet-50 model. Both models can be tested and the checkpoints are saved in the attached files.

## Deep Learning Project 3 (Encoder-Decoder Implementation):
The Encoder-Decoder Project tackled the creation of a model which classifies pixels of images in different categories. Here two different models were trained, the U-Net and the Res-Net. It works mainly with a down-path to reduce dimensions and then a corresponding up-path to increase dimensions again. In between the single layers in both paths both are connected through concatenation. One can switch between the models with changing import, loading of trained model and model creation.

## Math for AI Project 1 (Visualizing PCA):
This file contains a solution to a exam exercise in which we had to visualize a PCA of the Olivetti faces dataset. I completed the task in two ways. First I calculated the principal components and transformed the data back to the original bases to visualize the transformed faces in comparison to the original one. Secondly I plotted the data in a pairwise plot of each principle component to each other.
