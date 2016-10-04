import tensorflow as tf


class CNN:
    """
    Implements a Convolutional Neural Network in TensorFlow

    Constructor Arguments
    ---------------------

    configuration (dict):
        structure (list) - Defines the structure of the CNN
            "INPUT:h,w,d" - Input layer of size [h x w x d] (not flattened)

            "CONV:fh,fw,fd,k:s"
                - Convolutional layer with k filters (or features),
                  filter size of [fh x fw x fd], stride of [1, s, s, 1],
                  and SAME zero-padding

            "MAXPOOL:h,w:s"
                - Max pooling layer with subsample size of [h x w],
                  stride of [1, s, s, 1], and SAME zero-padding

            "FC:n" - Fully-connected layer with n hidden neurons
            "OUTPUT:n" - Output layer of size n

        input_reshape_function
            - User-defined function that reshapes
              flattened input into image-form

        activation - Activation function of neurons in network
        cost_function - Cost function of CNN
        optimizer - Optimizer of cost funnction
        learning_rate - Learning rate of optimizer
        num_epochs - Number of epochs to run
        batch_size - Size of training batches
        display_step - When to display logs per epoch step

    train_dataset (DataSet): Training data in the form of a DataSet object

    Example configuration:
    mnist_config = {
        "structure": ["INPUT:28,28,1",
                      "CONV:5,5,1,32:1",
                      "MAXPOOL:2,2:2",
                      "CONV:5,5,32,64:1",
                      "MAXPOOL:2,2:2",
                      "FC:1024",
                      "OUTPUT:10"],
        "input_reshape_function": lambda x: tf.reshape(x, [-1, 28, 28, 1])
        "activation": tf.nn.relu,
        "optimizer": tf.train.AdamOptimizer,
        "cost_function": tf.nn.softmax_cross_entropy_with_logits,
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "batch_size": 100,
        "display_step": 1
    }

    Usage:
    from tensorflow.examples.tutorials.mnist import input_data
    from cnn import CNN

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist_cnn = CNN(mnist_config, mnist.train)
    mnist_cnn.train()

    """
    def __init__(self, configuration, train_dataset):
        self.load_configuration(configuration)
        self.train_dataset = train_dataset

    def build_model(self):
        input_layer = self.structure[0]
        (input_height, input_width, input_depth) = tuple(map(int, input_layer.split(":")[1].split(",")))

        # Setup input
        self.x = tf.placeholder(tf.float32, shape=[None, input_height * input_width * input_depth])

        # Reshape flat input into image based on user-defined reshape func.
        self.x_image = self.input_reshape_function(self.x)

        output_layer = self.structure[-1]
        output_layer_size = int(output_layer.split(":")[1])
        self.y = tf.placeholder(tf.float32, shape=[None, output_layer_size])

        self.weights = []
        self.biases = []

        hidden_layer_structure = self.structure[1:-1]
        self.hidden_layers = []

        for index, layer in enumerate(hidden_layer_structure):
            print(layer)
            if "CONV" in layer:
                (_, filter_dim, stride) = layer.split(":")
                stride = int(stride)
                (filter_height, filter_width, filter_depth, num_filters) = tuple(map(int, filter_dim.split(",")))

                weights = self.create_weight_variable([filter_height, filter_width, filter_depth, num_filters])
                biases = self.create_bias_variable([num_filters])

                self.weights.append(weights)
                self.biases.append(biases)

                # Check if the conv layer is the first hidden layer
                if index == 0:
                    self.hidden_layers.append(self.activation(
                        self.conv2d(self.x_image, weights, [1, stride, stride, 1]) + biases))
                else:
                    self.hidden_layers.append(self.activation(
                        self.conv2d(self.hidden_layers[index-1], weights, [1, stride, stride, 1]) + biases))

            elif "MAXPOOL" in layer:
                (_, window_dim, stride) = layer.split(":")
                stride = int(stride)
                (height, width) = tuple(map(int, window_dim.split(",")))
                self.hidden_layers.append(
                    self.max_pool(
                        self.hidden_layers[index-1],
                        [1, height, width, 1],
                        [1, stride, stride, 1]
                    )
                )

            elif "FC" in layer:
                num_units = int(layer.split(":")[1])

                # Check if FC layer is the first layer after the input layer
                if index == 0:
                    prev_layer_dims = self.x_image.get_shape().dims
                else:
                    prev_layer_dims = self.hidden_layers[index - 1] \
                                        .get_shape().dims

                if len(prev_layer_dims) == 4:
                    (_, height, width, depth) = prev_layer_dims
                    prev_layer_size = int(height) * int(width) * int(depth)

                    # Flatten previous layer
                    prev_layer = tf.reshape(self.hidden_layers[index - 1],
                                            [-1, prev_layer_size])
                elif len(prev_layer_dims) == 2:
                    (_, prev_layer_size) = prev_layer_dims
                    prev_layer = self.hidden_layers[index - 1]

                weights = self.create_weight_variable(
                    [prev_layer_size, num_units])
                biases = self.create_bias_variable([num_units])

                self.weights.append(weights)
                self.biases.append(biases)

                self.hidden_layers.append(
                    self.activation(tf.matmul(prev_layer, weights) + biases)
                )

        last_hidden_layer = self.hidden_layers[-1]
        # Assuming the last hidden layer is fully-connected
        (_, last_hidden_layer_size) = last_hidden_layer.get_shape().dims
        last_hidden_layer_size = int(last_hidden_layer_size)

        output_layer_weights = self.create_weight_variable([last_hidden_layer_size, output_layer_size])
        output_layer_biases = self.create_bias_variable([output_layer_size])

        # Output prediction of model
        self.y_model = tf.matmul(last_hidden_layer, output_layer_weights) + output_layer_biases

        # Determine cost of model
        self.cost = tf.reduce_mean(self.cost_function(self.y_model, self.y))

        # Define training step
        self.train_step = self.optimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Determine correct predictions
        self.correct_prediction = tf.equal(tf.argmax(self.y_model, 1), tf.argmax(self.y, 1))

        # Define accuracy measure
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self):
        """ Handles CNN training """

        # Build CNN model
        self.build_model()
        print("Finished building CNN model")

        # Save TF Session to instance
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        for epoch in range(self.num_epochs):
            avg_cost = 0.0
            total_batch = int(self.train_dataset.num_examples / self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = self.train_dataset.next_batch(self.batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, cost = self.sess.run([self.train_step, self.cost],
                                        feed_dict={
                                            self.x: batch_x,
                                            self.y: batch_y
                                        })
                # Compute average loss
                avg_cost += cost / total_batch
                # Debugging purposes
                print("Mini-batch %d"%i, "cost=", "{:.9f}".format(cost))

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    def test(self, test_dataset):
        """ Evaluates the trained model on a test data set """
        return self.accuracy.eval(
            {self.x: test_dataset.features, self.y: test_dataset.labels},
            session=self.sess  # Use trained model
        )

    def create_weight_variable(self, shape):
        """ Initalize weights using truncated normal distribution """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def create_bias_variable(self, shape):
        """ Intialize neurons with slightly positive bias """
        return tf.constant(0.1, shape=shape)

    def conv2d(self, x, weights, strides):
        """ Compute 2D convolution of given x and weights (filter) """
        return tf.nn.conv2d(x, weights, strides=strides, padding="SAME")

    def max_pool(self, x, ksize, strides):
        """ Compute max-pooling "activations" """
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding="SAME")

    def avg_pool(self, x, ksize, strides):
        # TODO: Implement average pooling

    def load_configuration(self, config):
        """ Sets configuration variables """
        self.structure = config["structure"]
        self.input_reshape_function = config["input_reshape_function"]
        self.cost_function = config["cost_function"]
        self.activation = config["activation"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]

        self.num_epochs = config["num_epochs"]
        self.display_step = config["display_step"]
        self.batch_size = config["batch_size"]
