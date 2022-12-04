# Machine-learning-projects
University projects in the courses COSE474 Deep Learning and MATH292 Mathematics for AI from Korea University.
In every program the dataset still has to be loaded. Information is provided in the corresponding exercise sheet.

## Deep Learning Project 1 (MLP Implementation):
This project contains a two layered neural network from scratch. It can calculate a forecast for the input data (in this case CIFAR-10 dataset) and is able to be trained via backpropagation. The specific exercises and the model infrastructure can be looked up in the pdf-files. I found the original project at the Stanford website <a href="https://www.w3schools.com](http://cs231n.stanford.edu/">CS231n: Convolutional Neural Networks for Visual Recognition</a>.

## Deep Learning Project 2 (CNN Architecture Implementation):
This project treats a pytorch environment in which the CIFAR10 dataset should be used. First of all the so called VGG-16 model was used and afterwards we had to fill in the missing pieces of the ResNet-50 model. Both models can be tested and the checkpoints are saved in the attached files.

## Deep Learning Project 3 (Encoder-Decoder Implementation):
The Encoder-Decoder Project tackles the creation of a model which classifies pixels of images in different categories. Here two different models are trained, the U-Net and the Res-Net. It works mainly with a down-path to reduce dimensions and then a corresponding up-path to increase dimensions again. In between the single layers in both paths both are connected through concatenation. In the program we first load a trained model, then train it on our own and save the validation images and the updated parameters. One can switch between the models with changing import, loading of trained model and model creation. The currently uploaded files contain a realization of the training and you can see the results in history. But the output weights are not saved because of the size of the files.

## Math for AI Project 1 (Visualizing PCA):
This file contains a solution to a exam exercise in which we had to visualize a PCA of the Olivetti faces dataset. I completed the task in two ways. First I calculated the principal components and transformed the data back to the original bases to visualize the transformed faces in comparison to the original one. Secondly I plotted the data in a pairwise plot of each principle component to each other.

##Math for AI Presentation (Optimizer Comparison):
This program written for a university presentation tackles the challenge of comparing different optimizers with 3D-plots in a kind of race. You can put in every parameter and function for 5 different kinds of optimizers and run the plot with output pngs, gif and txt file with the used parameters.
To run the code you can install the used packages in a conda environment:
conda create -n optimizer python=3.8
conda activate optimizer
pip install -r requirements.txt
python main.py
I wrote the program on basis of <a href="https://github.com/Jaewan-Yun/optimizer-visualization">this code<\a> which used an old version of tensorflow for the gradient calculation. I used pytorch instead.
