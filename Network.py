#The Network class creates and optimizes a neural network given any matrix of
#inputs and matrix of outputs, along with the dimensions of the neural network
#and number of epochs. The Mnist class imports the Mnist dataset, which is a
#dataset of matrix representations of handwritten digits along with their labels.
#It then creates an instance of the Network class with the specifications needed.
#Finally, the Run module creates an instance of the Mnist class, which in turn
#creates the Network with the specified parameters. To customize the Network,
#one can change the parameters of Mnist instance the in the Run module. The
#meaning of each parameter is documented. In addition, the learning rate of the
#network can be adjusted in the Network class. If the Network does not reach a
#satisfactory accuracy, the learning rate can either be increased or decreased.
#I have found that the best values seem to be between 0.03 and 0.005.

###############################################################################
#Network
import numpy as np

# Network expects an input matrix, an output matrix, an integer for number of
# hidden layers, an integer for number of number of activations per hidden layer
# (keep in mind that the activations are all still matrices, but I've
# disregarded the number of rows becasue those are equal to the number of
# training examples), and an int for number of epochs the neural net should
# perform.
class Network:
    def __init__(self, input, output, input_test, output_test, nOfLayers, nOfActivations, epochs):
        self.input = input
        self.output = output
        self.input_test = input_test
        self.output_test = output_test
        self.nOfLayers = nOfLayers
        self.nOfActivations = nOfActivations
        self.epochs = epochs

        self.activations = []
        self.weights = []
        self.biases = []

        self.learningRate = 0.1
        self.batchsize = 64

    #sigmoid function and its derivative
    def sigmoid(self, x, derivative=False):
       if derivative:
           y = self.sigmoid(x)
           return y * (1 - y)
       return 1.0 / (1.0 + np.exp(-x))

    def create(self):
        print("Creating Network...")
        #create the activations
        self.activations.append(self.input)
        for i in range(self.nOfLayers):
            self.activations.append(np.zeros((self.input.shape[0], self.nOfActivations)))
        self.activations.append(self.output)

        #create the weights
        for i in range(len(self.activations)-1):
            self.weights.append(np.random.randn(self.activations[i].shape[1],
            self.activations[i+1].shape[1]))

        #create the biases
        for i in range(len(self.activations)-1):
            self.biases.append(np.zeros(self.activations[i+1].shape[1]))

    #every activation matrix a(L) is given by sigmoid(w(L)a(L-1) + b(L)), where
    #L is the index of the current layer, w is the weight matrix, and b is the
    #bias matrix.
    def feed_forward(self, input, weights, biases):
        z = []
        activations = []
        activation = input

        for i in range(self.nOfLayers+1):
            activations.append(activation)
            z.append(np.dot(activation, weights[i])+biases[i])
            activation = self.sigmoid(z[i])

        return activation, z, activations

    # evaluate sum of cost between all predictions and expected output and then
    # the sum of that cost between all training examples
    def evaluate_cost(self, y, output):
        cost = np.mean(np.sum(((y - output)) ** 2, axis=1))
        return cost

    # Backpropagation Algorithm
    def back_propagate(self, y, z, weights, activations, output):
        weightGradient = []
        biasGradient = []
        firstTime = False

        #derivative of Cost with respect to output. The d represents the
        #derivative term.
        d = 2 * (y - output)
        d = d / len(d)

        for i in range(self.nOfLayers,-1,-1):
            #derivative of activation with respect to z
            d = d * self.sigmoid(z[i], True)

            #derivative of z with respect to weight matrix
            weightGradient.append(np.dot(activations[i].transpose(), d))

            #derivative of z with respect to bias vector
            biasGradient.append(d.sum(axis=0))

            #derivative of z with respect to activation of previous layer
            d = np.dot(d, weights[i].transpose())

        return weightGradient, biasGradient


    # Updating weights and biases based on weight and bias gradients calculated
    # in backProp algorithm
    def update_everything(self, weightGradient, biasGradient, weights, biases):
        p1 = len(weights)-1
        p2 = len(biases)-1
        new_weights = weights
        new_biases = biases

        for i in range(len(weights)):
            new_weights[i] = new_weights[i] - (self.learningRate * weightGradient[p1 - i])

        for j in range(len(biases)):
            new_biases[j] = new_biases[j] - (self.learningRate * biasGradient[p2 - j])

        return new_weights, new_biases

    # returns a slice of the input and output matrices corresponding to the
    # batchsize specified
    def choose_batch(self, input, output, index):
        x = index*self.batchsize
        input_batch = input[x:x+self.batchsize]
        output_batch = output[x:x+self.batchsize]
        return input_batch, output_batch

    # putting it all together...
    def optimize(self):
        print("Optimizing...")
        weights = self.weights
        biases = self.biases
        trials = int(len(self.input) / self.batchsize)

        for i in range(self.epochs):
            index = 0
            for j in range(trials):
                input, output = self.choose_batch(self.input, self.output, index)
                y, z, activations = self.feed_forward(input, weights, biases)
                cost = self.evaluate_cost(y, output)
                weightGradient, biasGradient = self.back_propagate(y, z, weights, activations, output)
                weights, biases = self.update_everything(weightGradient, biasGradient, weights, biases)
                index+=1

            print("Cost-->" + str(i) + " epochs:", end =" ")
            print(cost)
        return weights, biases

    # returns accuracy from 0-1 and the number of correcly labeled images
    def compute_accuracy(self, input, y, weights, biases):
        output, _xx, _yy = self.feed_forward(input, weights, biases)
        correct = (np.argmax(output, axis=1) == np.argmax(y, axis=1))
        return correct.mean(), correct.sum()

    def evaluate_model(self, weights, biases):
        print("Evaluating Model...")
        self.accuracy, self.correct = self.compute_accuracy(self.input, self.output, weights, biases)
        print()
        print("Correctly Labeled Train Images: " + str(self.correct))
        print("Total number of Train Images: " + str(len(self.output)))
        print("Accuracy of Training Data: " + str(self.accuracy * 100) + "%")

        self.accuracy_test, self.correct_test = self.compute_accuracy(self.input_test, self.output_test, weights, biases)
        print()
        print("Correctly Labeled Test Images: " + str(self.correct_test))
        print("Total number of Test Images: " + str(len(self.output_test)))
        print("Accuracy of Test Data: " + str(self.accuracy_test * 100) + "%")
