import nn
import math

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        y = self.run(x)
        return -1 if nn.as_scalar(y) < 0 else 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        # Keep iterating until no errors
        while True:
            # Initialize flag for weight update
            weight_update = False
            
            # Iterate one pt at a time
            for x, y in dataset.iterate_once(1):
                # Check if prediction is incorrect
                if self.get_prediction(x) != nn.as_scalar(y):
                    # Update weight
                    self.w.update(x, nn.as_scalar(y))
                    weight_update = True
            
            # Terminate loop once no more updates are made
            if not weight_update:
                break




class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, input_size=1, hidden_size=512, output_size=1, learning_rate=0.01):
        # Initialize your model parameters here
        self.lr = learning_rate
        self.w1 = nn.Parameter(input_size, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        self.w2 = nn.Parameter(hidden_size, hidden_size)
        self.b2 = nn.Parameter(1, hidden_size)
        self.w3 = nn.Parameter(hidden_size, output_size)
        self.b3 = nn.Parameter(1, output_size)
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Compute input layer
        input_layer = nn.Linear(x, self.w1)

        # Use ReLU for 1st hidden layer
        hidden_layer_1 = nn.ReLU(nn.AddBias(input_layer, self.b1))

        # 2nd hidden layer
        hidden_layer_2 = nn.ReLU(nn.AddBias(nn.Linear(hidden_layer_1, self.w2), self.b2))

        # Output layer
        output_layer = nn.AddBias(nn.Linear(hidden_layer_2, self.w3), self.b3)
        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        # Compute square loss
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)

    def train(self, dataset, batch_size=200, num_epochs=10):
        """
        Trains the model.
        """
        batch_size = 50
        tolerance = 0.015  # Loss tolerance to stop training
        loss = float('inf')  # Initialize loss to a very large value

        while loss >= tolerance:  # Continue training until loss is below tolerance
            # Iterate over the dataset in batches of size batch_size
            for x, y in dataset.iterate_once(batch_size):
                # Compute the loss and gradients for current batch
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)

                # Update the parameters using the gradients and the learning rate
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)

                # Print the loss for monitoring purposes
                loss = nn.as_scalar(loss)
                print(loss)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = .1
        self.w1 = nn.Parameter(784, 250)
        self.b1 = nn.Parameter(1, 250)
        self.w2 = nn.Parameter(250, 120)
        self.b2 = nn.Parameter(1, 120)
        self.w3 = nn.Parameter(120, 64)
        self.b3 = nn.Parameter(1, 64)
        self.w4 = nn.Parameter(64, 10)
        self.b4 = nn.Parameter(1, 10)
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, 
                       self.w4, self.b4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.w1) # linear transformation using w1
        x = nn.AddBias(x, self.b1) # add b1 to output
        x = nn.ReLU(x) # apply ReLU

        x = nn.Linear(x, self.w2) # linear transformation using w2
        x = nn.AddBias(x, self.b2) # add b2 to output
        x = nn.ReLU(x) # apply ReLU

        x = nn.Linear(x, self.w3) # linear transformation using w3
        x = nn.AddBias(x, self.b3) # add b3 to output
        x = nn.ReLU(x) # apply ReLU

        x = nn.Linear(x, self.w4) # linear transformation using w4
        x = nn.AddBias(x, self.b4) # add b4 to output

        # return final output
        return x
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(x)
        return nn.SoftmaxLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Set the batch size
        batch_size = 100

        # Train for a fixed number of epochs
        for epoch in range(100):
            # Iterate through the training dataset
            for x, y in dataset.iterate_once(batch_size):
                # Calculate the loss and gradients
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)

                # Update the parameters using the gradients and learning rate
                for i, param in enumerate(self.params):
                    param.update(grads[i], -self.lr)

            # Calculate the validation accuracy
            accuracy = dataset.get_validation_accuracy()

            # Stop training if the validation accuracy is high enough
            if accuracy >= 0.98:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Define layer dimensions
        input_dim = self.num_chars
        hidden_dim = 100
        output_dim = 5

        # Initialize weights and biases
        self.W1 = nn.Parameter(input_dim, hidden_dim)
        self.b1 = nn.Parameter(1, hidden_dim)
        self.W2 = nn.Parameter(hidden_dim, hidden_dim)
        self.b2 = nn.Parameter(1, hidden_dim)
        self.W1_hidden = nn.Parameter(hidden_dim, hidden_dim)
        self.b1_hidden = nn.Parameter(1, hidden_dim)
        self.W2_hidden = nn.Parameter(hidden_dim, hidden_dim)
        self.b2_hidden = nn.Parameter(1, hidden_dim)
        self.W_final = nn.Parameter(hidden_dim, output_dim)
        self.b_final = nn.Parameter(1, output_dim)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W1_hidden, self.b1_hidden, 
                       self.W2_hidden, self.b2_hidden, self.W_final, self.b_final]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.W1), self.b1))
        for i in range(1, len(xs)):
            h = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(xs[i], self.W1), nn.Linear(h, self.W1_hidden)), self.b1_hidden))
        return nn.AddBias(nn.Linear(h, self.W_final), self.b_final)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(xs)
        return nn.SoftmaxLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.15
        for epoch in range(21):
            for x_batch, y_batch in dataset.iterate_once(60):
                grads = nn.gradients(self.get_loss(x_batch, y_batch), self.params)
                for i, param in enumerate(self.params):
                    param.update(grads[i], -learning_rate)