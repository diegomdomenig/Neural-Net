#Network class creates and optimizes a neural network given any matrix of inputs
#and matrix of outputs, along with the dimensions of the neural network and
#number of epochs. The Mnist class imports the Mnist dataset, which is a dataset
#of matrix representations of handwritten digits along with their labels. It
#then creates an instance of the Network class with the specifications needed.
#Finally, the Run module creates an instance of the Mnist class, which in turn
#creates the Network with the specified parameters. To customize the Network,
#one can change the parameters of Mnist instance the in the Run module. The
#meaning of each parameter is documented. In addition, the learning rate of the
#network can be adjusted in the Network class. If the Network does not reach a
#satisfactory accuracy, the learning rate can either be increased or decreased.
#I have found that the best values seem to be between 0.03 and 0.005. The
#README file has outlines the math used to approach a local minimum of the
#cost function.

from mlxtend.data import loadlocal_mnist
import numpy as np
import Network
import os

class Mnist():
    def __init__(self, nOfLayers=3, nOfActivations=20, epochs=100):
        self.nOfLayers = nOfLayers
        self.nOfActivations = nOfActivations
        self.epochs = epochs
        self.final_weights = []
        self.final_biases = []

    def load_data(self):
        #loading mnist training dataset
        print("Gathering Data...")
        path = os.path.dirname(os.path.realpath(__file__))
        self.X, self.y = loadlocal_mnist(
                images_path = path + '/train-images-idx3-ubyte',
                labels_path = path + '/train-labels-idx1-ubyte')
        self.X_test, self.y_test = loadlocal_mnist(
                images_path = path + '/t10k-images-idx3-ubyte',
                labels_path = path + '/t10k-labels-idx1-ubyte')


    def preprocess_data(self):
        #preprocessing the data to fit the prespecified parameters. In addition,
        #the labels of the images are simple integers, but I want each label to
        #be an array that indicates what that label is based on the index of the
        #1 in the array. So for example, if the label of the image is 5, i want
        #the new label to be the following array: label = [0,0,0,0,0,1,0,0,0,0]

        #self.input
        self.input = self.X

        #self.output --> converting the labels from a single integer like "5" to
        #an array of zeros with the number "1" at the index of the label's value
        #(in this case, the array would be [0 0 0 0 0 1 0 0 0 0])
        k = []
        for i in range(len(self.input)):
            a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            a[self.y[i]] = 1
            k.append(a)
        self.output = np.array(k)

        #self.input_test
        self.input_test = self.X_test

        #self.output_test
        k = []
        for i in range(len(self.input_test)):
            a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            a[self.y_test[i]] = 1
            k.append(a)
        self.output_test = np.array(k)


    def create_network(self):
        #(input, output, input_test, output_test, nOfLayers, nOfActivations, epochs)
        self.model = Network.Network(self.input, self.output, self.input_test,
            self.output_test, self.nOfLayers, self.nOfActivations, self.epochs)
        self.model.create()

    def optimize(self):
        self.final_weights, self.final_biases = self.model.optimize()

    def evaluate(self):
        self.model.evaluate_model(self.final_weights, self.final_biases)
