import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)
# Returns the ReLU value of the input x
def relu(x):
    return max(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (x>0).astype(int)

## TODO 1a: Return the sigmoid value of the input x
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

## TODO 1b: Return the derivative of the sigmoid value of the input x
def sigmoid_derivative(x):
    return sigmoid(x)*(1- sigmoid(x))

## TODO 1c: Return the derivative of the tanh value of the input x
def tanh(x):
    return (2*sigmoid(2*x) - 1)

## TODO 1d: Return the derivative of the tanh value of the input x
def tanh_derivative(x):
    return (1 - ((tanh(x))**2))

# Mapping from string to function
str_to_func = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}


# Given a list of activation functions, the following function returns
# the corresponding list of activation functions and their derivatives
def get_activation_functions(activations):  
    activation_funcs, activation_derivatives = [], []
    for activation in activations:
        activation_func, activation_derivative = str_to_func[activation]
        activation_funcs.append(activation_func)
        activation_derivatives.append(activation_derivative)
    return activation_funcs, activation_derivatives

class NN:
    def __init__(self, input_dim, hidden_dims, activations=None):
        '''
        Parameters
        ----------
        input_dim : int
            size of the input layer.
        hidden_dims : LIST<int>
            List of positive integers where each integer corresponds to the number of neurons 
            in the hidden layers. The list excludes the number of neurons in the output layer.
            For this problem, we fix the output layer to have just 1 neuron.
        activations : LIST<string>, optional
            List of strings where each string corresponds to the activation function to be used 
            for all hidden layers. The list excludes the activation function for the output layer.
            For this problem, we fix the output layer to have the sigmoid activation function.
        ----------
        Returns : None
        ----------
        '''
        assert(len(hidden_dims) > 0)
        assert(activations == None or len(hidden_dims) == len(activations))
         
        # If activations is None, we use sigmoid activation for all layers
        if activations == None:
            self.activations = [sigmoid]*(len(hidden_dims)+1)
            self.activation_derivatives = [sigmoid_derivative]*(len(hidden_dims)+1)
        else:
            self.activations, self.activation_derivatives = get_activation_functions(activations + ['sigmoid'])

        ## TODO 2: Initialize weights and biases for all hidden and output layers
        ## Initialization can be done with random normal values, you are free to use
        ## any other initialization technique.
        self.weights = [np.random.uniform(-np.sqrt(6/(input_dim + hidden_dims[0])), np.sqrt(6/(input_dim + hidden_dims[0])), (input_dim, hidden_dims[0]))]
        self.weights.extend([np.random.uniform(-np.sqrt(6/(hidden_dims[i-1] + hidden_dims[i])), np.sqrt(6/(hidden_dims[i-1] + hidden_dims[i])),  (hidden_dims[i-1], hidden_dims[i])) for i in range(1, len(hidden_dims))])
        self.weights.append(np.random.uniform(-np.sqrt(6/(hidden_dims[-1] + 1)), np.sqrt(6/(hidden_dims[-1] + 1)),(self.weights[-1].shape[1], 1)))
        self.biases = [np.random.uniform(-np.sqrt(6/(hidden_dims[i] + 1)), np.sqrt(6/(hidden_dims[i] + 1)), (1, hidden_dims[i])) for i in range(len(hidden_dims))]
        self.biases.append(np.random.uniform(-np.sqrt(3), np.sqrt(3), (1, 1)))
        self.initialized_zeros = False
        self.epochs = 0

    def forward(self, X):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        ----------
        Returns : output probabilities, numpy array of shape (N, 1) 
        ----------
        '''
        # Forward pass
        
        
        self.realized_activations = []
        self.dragon_balls = []
        self.realized_activations.append(X)
        self.dragon_balls.append("Z-eno")
        ## TODO 3a: Compute activations for all the nodes with the corresponding
        ## activation function of each layer applied to the hidden nodes
        for i in range(len(self.weights)):
            z = (np.matmul(self.realized_activations[-1], self.weights[i]) + self.biases[i])
            self.dragon_balls.append(z)
            self.realized_activations.append(self.activations[i](z))
        
        ## TODO 3b: Calculate the output probabilities of shape (N, 1) where N is number of examples
    

        return self.realized_activations[-1]







    def backward(self, X, y):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        y : target labels, numpy array of shape (N, 1) where N is the number of examples
        ----------
        Returns : gradients of weights and biases
        ----------
        '''
        # Backpropagation

        ## TODO 4a: Compute gradients for the output layer after computing derivative of 
        ## sigmoid-based binary cross-entropy loss
        ## Hint: When computing the derivative of the cross-entropy loss, don't forget to 
        ## divide the gradients by N (number of examples)  
        y = np.reshape(y, (y.shape[0], 1))
        enn = y.shape[0]
        delta_activation = [np.random.rand(enn, item.shape[1]) for item in self.weights]
        assert(delta_activation[-1].shape[1] == 1 and delta_activation[-1].shape[0] == enn)

        delta_activation[-1] = ((np.divide((1 - y), (1 - self.realized_activations[-1])) - np.divide(y, self.realized_activations[-1]))*self.activation_derivatives[-1](self.realized_activations[-1]))*(1/enn)  # this assumes the final activation is a sigmoid
        # print(delta_activation[-1].shape)
        # time.sleep(3)

        ## TODO 4b: Next, compute gradients for all weights and biases for all layers
        ## Hint: Start from the output layer and move backwards to the first hidden layer

        self.grad_weights = [[i] for i in range(len(self.weights))]
        self.grad_biases = [[i] for i in range(len(self.biases))]
        for i in reversed(range(len(self.weights))):        
            if(i > 0):
                delta_activation[i-1] = np.matmul(delta_activation[i], self.weights[i].T)
                delta_activation[i-1] *= self.activation_derivatives[i-1](self.dragon_balls[i])
            self.grad_weights[i] = np.matmul(self.realized_activations[i].T, delta_activation[i])
            self.grad_biases[i] = np.sum(delta_activation[i], axis = 0, keepdims=True)

        return self.grad_weights, self.grad_biases
    





    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                gd_flag: 1 for Vanilla GD, 2 for GD with Exponential Decay, 3 for Momentum
                momentum: Momentum coefficient, used when gd_flag is 3.
                decay_constant: Decay constant for exponential learning rate decay, used when gd_flag is 2.
            epoch: Current epoch number
        '''
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        decay_constant = optimizer_params['decay_constant']

        for i in range(len(weights)):
            assert(weights[i].shape == delta_weights[i].shape)
        ### Calculate updated weights using methods as indicated by gd_flag

        ## TODO 5a: Variant 1(gd_flag = 1): Vanilla GD with Static Learning Rate
        ## Use the hyperparameter learning_rate as the static learning rate
        updated_W = []
        updated_B = []
        if(gd_flag == 1):
            for i in range(len(weights)):
                updated_W.append(weights[i] - learning_rate*delta_weights[i])
                updated_B.append(biases[i] - learning_rate*delta_biases[i])

        ## TODO 5b: Variant 2(gd_flag = 2): Vanilla GD with Exponential Learning Rate Decay
        ## Use the hyperparameter learning_rate as the initial learning rate
        ## Use the parameter epoch for t
        ## Use the hyperparameter decay_constant as the decay constant
        elif(gd_flag == 2):
            adjusted_learning_rate = learning_rate*(np.exp((-decay_constant)*epoch).item())
            for i in  range(len(weights)):
                updated_W.append(weights[i] - adjusted_learning_rate*delta_weights[i])
                updated_B.append(biases[i] - adjusted_learning_rate*delta_biases[i])
        ## TODO 5c: Variant 3(gd_flag = 3): GD with Momentum
        ## Use the hyperparameters learning_rate and momentum
        else:
            one_minus_momentum = 1 - momentum
            if self.initialized_zeros == False:
                self.grad_w_zero = [item for item in delta_weights]
                self.grad_b_zero = [item for item in delta_biases]
                self.initialized_zeros = True
            for i in range(len(weights)):
                momentum_delta_w = delta_weights[i]*one_minus_momentum + (1 - one_minus_momentum)*(self.grad_w_zero[i])
                momentum_delta_b = delta_biases[i]*one_minus_momentum + (1 - one_minus_momentum)*(self.grad_b_zero[i])
                updated_W.append(weights[i] - learning_rate*momentum_delta_w)
                updated_B.append(biases[i] - learning_rate*momentum_delta_b)
                self.grad_w_zero[i] = momentum_delta_w
                self.grad_b_zero[i] = momentum_delta_b
        
        return updated_W, updated_B

    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                beta: Exponential decay rate for the first moment estimates.
                gamma: Exponential decay rate for the second moment estimates.
                eps: A small constant for numerical stability.
        '''
        learning_rate = optimizer_params['learning_rate']
        beta = optimizer_params['beta']
        gamma = optimizer_params['gamma']
        eps = optimizer_params['eps']       

        updated_W = []
        updated_B = []
        ## TODO 6: Return updated weights and biases for the hidden layer based on the update rules for Adam Optimizer
        if self.initialized_zeros == False:
            self.grad_w_zero = [item for item in delta_weights]
            self.grad_b_zero = [item for item in delta_biases]
            self.s_w = [np.zeros((1, item.shape[1])) for item in delta_weights] # one normalizer for every neuron in the next layer, then that for every layer
            self.s_b = [0 for item in delta_biases] # one normalizer term for every layer_bias vector
        for i in range(len(weights)):
            momentum_delta_w = delta_weights[i]*(1 - beta) + beta*(self.grad_w_zero[i])
            momentum_delta_b = delta_biases[i]*(1 - beta) + beta*(self.grad_b_zero[i])
            self.grad_w_zero[i] = np.atleast_2d(momentum_delta_w)
            self.grad_b_zero[i] = np.atleast_2d(momentum_delta_b)
            column_wise_norms = (np.atleast_2d([np.dot(np.atleast_2d(item), np.atleast_2d(item).T).item() for item in delta_weights[i].T]))
            bias_norm = np.dot(delta_biases[i], delta_biases[i].T).item()
            self.s_w[i] = self.s_w[i]*(gamma) + (1 - gamma)*(column_wise_norms)
            self.s_b[i] = self.s_b[i]*(gamma) + (1 - gamma)*(bias_norm)
            del_w_norm = np.sqrt(self.s_w[i]/(1 - (gamma**self.epochs))) + eps
            del_b_norm = np.sqrt(self.s_b[i]/(1 - (gamma**self.epochs))) + eps
            adjusted_w_velocity_term = (momentum_delta_w/(1 - (beta**self.epochs)))/(del_w_norm)
            adjusted_b_velocity_term = (momentum_delta_b/(1 - (beta**self.epochs)))/(del_b_norm)
            # print(adjusted_velocity_term)
            # time.sleep(10)
            updated_W.append(weights[i] - learning_rate*adjusted_w_velocity_term)
            updated_B.append(biases[i] - learning_rate*adjusted_b_velocity_term)
        return updated_W, updated_B



    def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            # Divide X,y into batches
            X_batches = np.array_split(X_train, X_train.shape[0]//batch_size)
            y_batches = np.array_split(y_train, y_train.shape[0]//batch_size)
            self.epochs += 1
            for X, y in zip(X_batches, y_batches):
                # Forward pass
                self.forward(X)
                # Backpropagation and gradient descent weight updates
                dW, db = self.backward(X, y)
                if optimizer == "adam":
                    self.weights, self.biases = self.step_adam(
                        self.weights, self.biases, dW, db, optimizer_params)
                elif optimizer == "bgd":
                    self.weights, self.biases = self.step_bgd(
                        self.weights, self.biases, dW, db, optimizer_params, epoch)

            # Compute the training accuracy and training loss
            train_preds = self.forward(X_train)
            train_loss = np.mean(-y_train*np.log(train_preds) - (1-y_train)*np.log(1-train_preds))
            train_accuracy = np.mean((train_preds > 0.5).reshape(-1,) == y_train)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            train_losses.append(train_loss)

            # Compute the test accuracy and test loss
            test_preds = self.forward(X_eval)
            test_loss = np.mean(-y_eval*np.log(test_preds) - (1-y_eval)*np.log(1-test_preds))
            test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)

        return train_losses, test_losses

    
    # Plot the loss curve
    def plot_loss(self, train_losses, test_losses, optimizer, optimizer_params):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if optimizer == "bgd":
            plt.savefig(f'loss_bgd_  {optimizer_params["gd_flag"]}  .png')
        else:
            plt.savefig(f'loss_adam.png')
 

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)
    # np.random.shuffle(data)
    # np.random.shuffle(data_eval)

    # Separate the data into X (features) and y (target) arrays
    X_train = data[:, :-1]
    y_train = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X_train.shape[1]
    X_train = X_train**2
    X_eval = X_eval**2
    hidden_dims = [4,2] # the last layer has just 1 neuron for classification
    num_epochs = 30
    batch_size = 100
    activations = ['sigmoid', 'sigmoid']
    optimizer = "bgd"
    optimizer_params = {
        'learning_rate': 0.1,
        'gd_flag': 2,
        'momentum': 0.9,
        'decay_constant': 0.1
    }
    
    # For Adam optimizer you can use the following
    # optimizer = "adam"
    # optimizer_params = {
    #     'learning_rate': 0.01,
    #     'beta' : 0.92,
    #     'gamma' : 0.99,
    #     'eps' : 1e-8
    # }

     
    model = NN(input_dim, hidden_dims)
    print(model.weights)
    train_losses, test_losses = model.train(X_train, y_train, X_eval, y_eval,
                                    num_epochs, batch_size, optimizer, optimizer_params) #trained on concentric circle data 
    print(model.weights)
    test_preds = model.forward(X_eval)
    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    model.plot_loss(train_losses, test_losses, optimizer, optimizer_params)


# alhamdulilah -> thank you god
# avzaldulilah -> in gods hands // likely incorrect translation
# astagfidullah -> forgive me god 
# mashallah -> as god wills 
# inshallah -> if god wills 
