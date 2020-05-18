import h5py 
import numpy as np 
import matplotlib.pyplot as plt

class l_layer_network:
    """
    this network will work a L-layer network such that layers 1 to L-1 use the relu activation function, and the output layer uses a sigmoid activation function. 
    (should work for output neurons > 1 as well)
    """
    def __init__(self,layers_dims,training_data_x,training_data_y,learning_rate,iterations):
        self.layers = layers_dims
        self.num_layers = len(layers_dims)
        self.X = training_data_x # (size: (num_features,num_samples))
        self.m = self.X.shape[1]
        self.Y = training_data_y # (size: (1,num_samples))
        self.alpha = learning_rate
        self.parameters = {
            'W1': np.random.randn(layers_dims[0],training_data_x.shape[0])*0.01,
            'b1': np.zeros((layers_dims[0],1)) # will be broadcasted to size: (neurons,num_samples)
            }
        self.num_iter = iterations
        for i,neurons in enumerate(layers_dims[1:],2):
            self.parameters['W' + str(i)] = np.random.randn(neurons,layers_dims[i-2])*0.01
            self.parameters['b' + str(i)] = np.zeros((neurons,1)) # will be broadcasted 
    
    def initialize_cache(self):
        self.A = [self.X]
        self.grads = {}
        self.caches = {}

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(z))

    @staticmethod
    def relu(z):
        return z * (z > 0)

    def forward_propagate(self):
        for i,layer in enumerate(self.layers,1):
            z = np.dot(self.parameters['W' + str(i)],self.A[i-1]) + self.parameters['b' + str(i)]
            a = self.sigmoid(z) if i == self.num_layers else self.relu(z)
            self.caches['cache' + str(i)] =  z
            self.A.append(a)
        return self.A[-1] # returns yhat 
    
    def compute_cost(self):
        return (-1/ self.m) * np.sum(np.multiply(self.Y, np.log(self.A[-1])) + np.multiply((1-self.Y), np.log(1-self.A[-1])))
    
    def backward_propagate(self):
        self.grads['dA' + str(self.num_layers)] = - (np.divide(self.Y, self.A[-1]) - np.divide(1 - self.Y, 1 - self.A[-1]))
        for i,layer in reversed(list(enumerate(self.layers,1))):
            if i == self.num_layers:
                self.grads['dZ' + str(i)] = np.array(self.grads['dA' + str(i)],copy=True)
                self.grads['dZ' + str(i)][self.caches['cache' + str(i)] <= 0 ] = 0 
            else:
                s = 1/(1+np.exp(self.caches['cache' + str(i)]))
                self.grads['dZ' + str(i)] = self.grads['dA' + str(i)]*s*(1-s)
            self.grads['dW'+ str(i)] = (1 / self.m) * np.dot(self.grads['dZ' + str(i)], self.A[i-1].T)
            self.grads['db' + str(i)] = (1 / self.m) * np.sum(self.grads['dZ' + str(i)], axis=1, keepdims=True) 
            self.grads['dA' + str(i-1)] = np.dot(self.parameters['W' + str(i)].T,self.grads['dZ' + str(i)])           

    def update_parameters(self):
        for i in range(self.num_layers):
            self.parameters['W' + str(i+1)] -= self.alpha*self.grads['dW' + str(i+1)]
            self.parameters['b' + str(i+1)] -= self.alpha*self.grads['db' + str(i+1)]
            
    
    def train_model(self):
        costs = []
        for i in range(self.num_iter):
            self.initialize_cache()
            self.forward_propagate()
            costs.append(self.compute_cost())
            self.backward_propagate()
            self.update_parameters()
            if (i % 100 == 0): print('cost after iteration {} is: {}'.format(i,self.compute_cost()))
        print('done training model')

    def predict(self,test_X,test_Y):
        self.X = test_X
        self.initialize_cache()
        predictions = self.forward_propagate()
        predictions = (predictions >= 0.5)
        assert predictions.shape == test_Y.shape
        accuracy = np.equal(predictions,test_Y)
        accuracy_pct = np.sum(accuracy)*100/predictions.shape[1]
        return accuracy_pct, predictions


def load_dataset():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

num_samples = train_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_X = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T/255
test_X = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T/255


net = l_layer_network([20,7,5,1],train_X,train_set_y,0.75,2000)

net.train_model()
pct,predictions = net.predict(train_X,train_set_y)
print(predictions,'the percent accuracy is : {}'.format(pct))