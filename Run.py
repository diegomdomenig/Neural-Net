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

import Mnist

#(nOfLayers, nOfActivations, epochs)
model = Mnist.Mnist(3, 40, 200)
model.load_data()
model.preprocess_data()
model.create_network()
model.optimize()
model.evaluate()
