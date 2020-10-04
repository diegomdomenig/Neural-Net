# Neural-Net
A Dense Neural Net implemented from scratch with Python 3

## Prerequisites
- python 3
- numpy `pip3 install numpy`
- mlxtend `pip3 install mlxtend`

## Downloading the Images
In order to test out the Neural Net on the MNIST Dataset, you will have to download the images and labels here. Make sure to keep the file names as they are and add them to your your project's directory

- [Training Set Images](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
- [Training Set Labels](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
- [Test Set Images](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
- [Test Set Labels](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

## Running the Program
To run the network, set the nOfLayers, nOfActivations, and epochs you want your network to have in the `Run.py` file. Then, in the root directory of your project, run `python3 Run.py` and the network should start training.
