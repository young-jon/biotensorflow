# TODO:  modify class DA to initialize to pretrain weights
# TODO:  test save_model and restoring model after saving multiple different architectures in same directory
# TODO:  other types of weight intiialization (see deep learning tutorials)
# TODO:  in save_model, also need to save config and paths to file
# TODO:  better docstring---description of parameters
# TODO:  regularization
# TODO:  testing

from __future__ import division, print_function, absolute_import
import time
import csv
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import get_image_dims

class DA(object):
    '''
    A Deep Autoencoder (also know as: finetuning) implementation using the 
    TensorFlow library. See __main__ for example usage.

    Args:

    sess (Session or InteractiveSession): a session
    config (dict): file of hyperparameters
    train_dataset (DataSet): Training data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector.
    validation_dataset (DataSet): same as train_dataset

    '''

    def __init__(self, sess, config, train_dataset, validation_dataset, 
                 pretrain_weights=None, pretrain_biases=None):
        '''Same as DNN except for encoder_hidden_layers'''
        self.sess = sess
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.pretrain_weights = pretrain_weights
        self.pretrain_biases = pretrain_biases
        self.save_path = config['save_path']
        self.encoder_hidden_layers = config['encoder_hidden_layers']
        self.activation = config['activation']
        self.cost_function = config['cost_function']
        self.optimizer = config['optimizer']
        self.regularizer = config['regularizer']
        self.learning_rate = config['learning_rate']
        self.training_epochs = config['training_epochs']
        self.batch_size = config['batch_size']
        self.display_step = config['display_step']
        self.save_costs_to_csv = config['save_costs_to_csv']

        self._build_graph()

    def _build_graph(self):  ### Different from DNN
        '''Builds the DA graph. This function is intended to be called by __init__'''
        ### builds a symbolic deep autoencoder graph based on the config hyperparameters
        print('Building Graph...')
        self.n_input = self.train_dataset.features.shape[1] # determine from train dataset

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])

        # create a list of the sizes of all hidden layers (encoder and decoder)
        self.hidden_layers = self.encoder_hidden_layers[:]
        for h in reversed(self.encoder_hidden_layers[:-1]):
            self.hidden_layers.append(h)

        # Store layer weights & biases (initialized using random_normal)
        all_layers = [self.n_input] + self.hidden_layers + [self.n_input]
        print('Network Architecture: ', all_layers)
        self.weights=[]
        self.biases=[]
        if self.pretrain_weights and self.pretrain_biases:
            print('Using pretrained weights and biases.')
            for i in range(len(all_layers)-1):
                self.weights.append(tf.Variable(self.pretrain_weights[i]))
                self.biases.append(tf.Variable(self.pretrain_biases[i]))
        else:
            for i in range(len(all_layers)-1):
                self.weights.append(tf.Variable(tf.random_normal([all_layers[i], all_layers[i+1]])))
                self.biases.append(tf.Variable(tf.random_normal([all_layers[i+1]])))

        # CREATE MODEL
        # create hidden layer 1
        self.model = []
        self.model.append(self.activation(tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0])))
        # create remaining hidden layers
        for j in range(len(self.hidden_layers))[1:]:
            self.model.append(self.activation(tf.add(tf.matmul(self.model[j-1], self.weights[j]), self.biases[j])))
        #create output layer
        self.model.append(tf.add(tf.matmul(self.model[-1], self.weights[-1]), self.biases[-1])) 

        # Construct model
        self.logits = self.model[-1]  ### output layer logits

        ### NOTES ###
        # the output of tf.nn.sigmoid_cross_entropy_with_logits(logits, y) is a 2D matrix (256,784) 
        # of size (minibatch rows, # of logits). each entry in the matrix is the cross-entropy 
        # (scalar value) for a single example for one of the 784 outputs. each row represents 784 
        # cross-entropy values for a single example. tf.reduce_mean calculates the mean of this array 
        # along all dimensions (by default), meaning it reduces a matrix of values to a single scalar 
        # (or in the case of softmax_cross_entropy_with_logits, reduces an array to a single scalar).
        # Therefore, the cost variable below is a scalar value. (the average cost for a minibatch
        # or the average cost across of 256x784 cross-entropy error values). 
        # see tf.nn.sigmoid_cross_entropy_with_logits??

        # Define cost (objective function) and optimizer
        self.cost = tf.reduce_mean(self.cost_function(self.logits, self.x))
        # MSE (below) seems to give worse results than cross entropy
        # self.cost = tf.reduce_mean(tf.pow(self.x - tf.nn.sigmoid(self.logits), 2))
        self.train_step = self.optimizer(learning_rate=self.learning_rate).minimize(self.cost)
        print('Finished Building Autoencoder Graph')

    def train(self):   ### Same as DNN except for feed_dict
        print('Training Autoencoder...')
        # initialize containers for writing results to file
        self.train_cost = []; self.validation_cost = []; 

        # Initializing the variables
        init = tf.initialize_all_variables()

        # 'Saver' op to save and restore all the variables
        # see https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
        self.saver = tf.train.Saver()

        # Launch the graph
        # with tf.Session() as self.sess:
        self.sess.run(init)

        # Training cycle
        for epoch in range(self.training_epochs):
            total_cost = 0.
            total_batch = int(self.train_dataset.num_examples/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = self.train_dataset.next_batch(self.batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.train_step, self.cost], feed_dict={self.x: batch_x})
        
                # Collect cost for each batch
                total_cost += c

            # Compute average loss for each epoch
            avg_cost = total_cost/total_batch

            #compute validation set average cost for each epoch given current state of weights
            validation_avg_cost = self.cost.eval({self.x: self.validation_dataset.features})

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "train cost=", \
                    "{:.9f}".format(avg_cost), "validation cost=", \
                    "{:.9f}".format(validation_avg_cost))

            #collect costs to save to file
            self.train_cost.append(avg_cost)
            self.validation_cost.append(validation_avg_cost)
        print("Optimization Finished!")

        if self.save_costs_to_csv:
            self.save_train_and_validation_cost()

    def save_train_and_validation_cost(self):
        '''Saves average train and validation set costs to .csv, all with unique filenames.
        Exactly the same as DNN version.'''
        # write validation_cost to its own separate file
        name = 'validation_costs_'
        file_path = self.save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self.validation_cost)
        # all error measures in one file
        df_to_disk = pd.DataFrame([self.train_cost, self.validation_cost],
                                    index=[[self.hidden_layers,self.learning_rate,
                                            self.training_epochs,self.batch_size], ''])
        df_to_disk['error_type'] = ['train_cost', 'validation_cost']
        # create file name and save as .csv
        name = 'all_costs_'
        file_path = self.save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
        df_to_disk.to_csv(file_path)
        print("Train and validation costs saved in file: %s" % file_path)

    def save_model(self, file_name):
        ### SAVE MODEL WEIGHTS TO DISK
        file_path = self.saver.save(self.sess, self.save_path + file_name)
        print("Model saved in file: %s" % file_path)

    def get_reconstruction_images(self, output_layer_activation, num_images=10, color='magma'):
        '''dispay 10 images from validation set and their reconstructions'''
        dims = get_image_dims(self.n_input)
        encode_decode = self.sess.run(output_layer_activation(self.logits), 
                            feed_dict={self.x: self.validation_dataset.features[:num_images]})
        ### Compare original images with their reconstructions
        f, a = plt.subplots(2, num_images, figsize=(40, 3))
        for i in range(num_images):
            a[0][i].imshow(np.reshape(self.validation_dataset.features[i], dims), cmap=color)
            a[1][i].imshow(np.reshape(encode_decode[i], dims), cmap=color)
        f.show()
        plt.draw()
        plt.waitforbuttonpress()
        return encode_decode



### EXAMPLE USAGE
if __name__ == '__main__':
    from deep_autoencoder import DA
    import tensorflow as tf
    import numpy as np
    from dataset import DataSet
    from utils import read_data_file, sep_data_train_test_val

    ### use non-mnist dataset
    f = read_data_file('/Users/jdy10/Code/python/biotensorflow/data_400.csv')
    # f = f[:,0:400]
    # np.savetxt("data_400.csv", f, delimiter=",") # save numpy array as .csv
    data = sep_data_train_test_val(f,0.9,0.05,0.05)  # should really use (0.7,0.15,0.15)
    train_dataset = data['train']
    test_dataset = data['test']
    validation_dataset = data['validation']

    ### uncomment below to use MNIST
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    # train_dataset = DataSet(mnist.train.images, mnist.train.labels)
    # validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)

### SETUP NEURAL NETWORK HYPERPARAMETERS
    config = {
        'save_path': '/Users/jdy10/Output/biotensorflow/',
        'encoder_hidden_layers': [100],
        'activation': tf.nn.sigmoid,
        #tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, tf.nn.elu
        'cost_function': tf.nn.sigmoid_cross_entropy_with_logits,
        'optimizer': tf.train.AdamOptimizer,
        #AdadeltaOptimizer,AdagradOptimizer,AdamOptimizer,MomentumOptimizer,RMSPropOptimizer,GradientDescentOptimizer
        'regularizer': None,
        'learning_rate': 0.009,  
        'training_epochs': 300,
        'batch_size': 100,
        'display_step': 1,
        'save_costs_to_csv': True
    }

    ### Buid, train, and save model
    with tf.Session() as sess:
        da = DA(sess, config, train_dataset, validation_dataset)  # init config and build graph
        da.train() 
        da.save_model('model.ckpt')
        rec=da.get_reconstruction_images(tf.nn.sigmoid, num_images=15) # argument here is the output layer activation
        # function for generating images of the reconstructions. This should match the activation 
        # used in the cost_function. Use tf.identity to output affine transformation without an 
        # activation function.  


    ### To reload model in same ipython session with same graph defined, run:
    # sess = tf.InteractiveSession()
    # saver = tf.train.Saver()
    # saver.restore(sess, config['save_path'] + 'model.ckpt')

    ### To reload model in new ipython session see random.py

    ### example usage after model loaded 
    # weights[0].eval()  # print 1st layer weights
    # h1=model[0].eval({x: train_dataset})  # get hidden layer 1 values using train data
    ### save hidden layer values
    # import numpy as np
    # np.savetxt(save_path + 'h1_train.csv', h1, delimiter=",")  ### np.loadtxt('h1_train.csv', delimiter=",")

