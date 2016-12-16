from __future__ import division, print_function, absolute_import
from deep_autoencoder import DA
import tensorflow as tf
import numpy as np
from dataset import DataSet
from tensorflow.examples.tutorials.mnist import input_data
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_image_dims

# TODO:  implement pseudolikelihood (see scikit, deep learning tutorials)
# TODO:  other measures of generalization error
# TODO:  automatic differentiation version


class RBM(object):
    '''
    A Restricted Boltzmann Machine implementation using 1-step of Contrastive 
    Divergence and the TensorFlow library. See __main__ for example usage. 
    This class includes 3 versions of a Restricted Boltzmann Machine. They are 
    called 'Hinton_2006', 'Ruslan_new', and 'Bengio'. You designate the version 
    with an argument in config. These 3 versions differ based on whether 
    sampling or probabilities are used for different calculations in Contrastive 
    Divergence and the resulting weight and biases updates. The reconstructions 
    in 'Hinton_2006' are probabilities, while the reconstructions in 
    'Ruslan_new' and 'Bengio' are binary (i.e. sampled from the reconstruction 
    probability distribution). I recommend using 'Ruslan_new' if you want your
    reconstructions to be binary. 
    
    'Hinton_2006' is based on code from Hinton and Salakhutdinov's 2006 Science 
    paper (http://www.cs.toronto.edu/~hinton/code/rbm.m). 'Ruslan_new' is based 
    on code from http://www.cs.toronto.edu/~rsalakhu/code_DBM/rbm.m. 'Bengio' is 
    based on pseudocode from page 33 of 
    http://www.iro.umontreal.ca/~bengioy/papers/ftml.pdf

    This class expects your data to be binary or continuous between 0 and 1.
    If your data is continuous (gaussian) and outside the range [0,1], remove 
    the sigmoids (tf.nn.sigmoid) from contrastive divergence. 

    Args:

    sess (Session or InteractiveSession):  a session
    config (dict):  file of hyperparameters
    train_dataset (DataSet):  Training data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector. Data should be binary or 
        continuous between 0 and 1.
    validation_dataset (DataSet):  same as train_dataset

    Config dictionary

    save_path (str):  path to directory for saving
    rbm_hidden_layer (int):  size of hidden layer
    regularizer (None):  to be implemented
    learning_rate (float):  learning rate to use for updating weights and biases
    initial_momentum (float):  momentum to use for epochs 1-5
    final_momentum (float):  momentum to use after epoch 5
    weight_cost (float):  weight decay rate 
    training_epochs (int):  number of epochs for rbm training
    batch_size (int):  number of examples to use in each mini-batch
    display_step (int):  how frequently to print results to screen
    rbm_version (str):  'Hinton_2006', 'Ruslan_new', or 'Bengio'
    save_costs_to_csv (bool):  if True, saves error values to a file

    '''

    def __init__(self, sess, config, train_dataset, validation_dataset):
        '''Same as DNN except for encoder_hidden_layers'''
        self.sess = sess
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.save_path = config['save_path']
        self.rbm_hidden_layer = config['rbm_hidden_layer']
        self.regularizer = config['regularizer']
        self.learning_rate = config['learning_rate']
        self.initial_momentum = config['initial_momentum']
        self.final_momentum = config['final_momentum']
        self.weight_cost = config['weight_cost']
        self.training_epochs = config['training_epochs']
        self.batch_size = config['batch_size']
        self.display_step = config['display_step']
        self.save_costs_to_csv = config['save_costs_to_csv']
        self.rbm_version = config['rbm_version']
        assert self.rbm_version in ['Hinton_2006','Ruslan_new','Bengio'], \
            'Please use a valid rbm_version in config --- Hinton_2006, Ruslan_new, or Bengio'

        self._build_graph()

    def _build_graph(self):  
        '''Builds the RBM graph. This function is intended to be called by __init__'''
        ### builds a symbolic deep autoencoder graph based on the config hyperparameters
        self.start = time.time()
        print('Building Graph...')
        self.n_input = self.train_dataset.features.shape[1] # determine from train dataset

        # Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        sym_batch_size = tf.to_float(tf.shape(self.x)[0])  ### symbolic, can pass in data with diff # examples
        print('RBM Network Architecture: ', [self.n_input,self.rbm_hidden_layer,self.n_input])

        # Graph Hyperparameters
        self.lr = tf.constant(self.learning_rate, dtype=tf.float32)
        self.weight_c = tf.constant(self.weight_cost, dtype=tf.float32)
        self.momentum = tf.placeholder(tf.float32)

        # Graph Variables
        ### This initialization performed much better than just default tf.random_normal for weights and biases
        self.weights = tf.Variable(tf.random_normal([self.n_input, self.rbm_hidden_layer], stddev=0.1)) 
        # remove .1 to use mean 0 stddev 1. could also use stddev = 0.01 or 0.001.
        self.hid_biases = tf.Variable(tf.zeros([self.rbm_hidden_layer]))
        self.vis_biases = tf.Variable(tf.zeros([self.n_input]))

        weights_increment  = tf.Variable(tf.zeros([self.n_input, self.rbm_hidden_layer]))
        vis_bias_increment = tf.Variable(tf.zeros([self.n_input]))
        hid_bias_increment = tf.Variable(tf.zeros([self.rbm_hidden_layer]))

        ### BUILD MODEL
        ### Start positive phase of 1-step Contrastive Divergence
        self.pos_hid_probs = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.weights), self.hid_biases))
        pos_associations = tf.matmul(tf.transpose(self.x), self.pos_hid_probs)
        pos_hid_act = tf.reduce_sum(self.pos_hid_probs, 0)
        pos_vis_act = tf.reduce_sum(self.x, 0)

        ### Start negative phase
        ### Sample the hidden unit states {0,1} using distribution determined by self.pos_hid_probs
        self.pos_hid_sample = self.pos_hid_probs > tf.random_uniform(tf.shape(self.pos_hid_probs), 0, 1)
        self.pos_hid_sample = tf.to_float(self.pos_hid_sample)

        self.neg_vis_probs=tf.nn.sigmoid(tf.add(tf.matmul(self.pos_hid_sample,tf.transpose(self.weights)),self.vis_biases))

        if self.rbm_version == 'Hinton_2006':
            print('Using rbm_verison:  Hinton_2006')
            neg_hid_probs = tf.nn.sigmoid(tf.add(tf.matmul(self.neg_vis_probs, self.weights), self.hid_biases))
            neg_associations = tf.matmul(tf.transpose(self.neg_vis_probs), neg_hid_probs) 
            neg_hid_act = tf.reduce_sum(neg_hid_probs, 0)
            neg_vis_act = tf.reduce_sum(self.neg_vis_probs, 0)
            ### Calculate mean squared error
            self.mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - self.neg_vis_probs), 1))

        elif self.rbm_version in ['Ruslan_new', 'Bengio']:
            # Sample the visible unit states {0,1} using distribution determined by self.neg_vis_probs
            self.neg_vis_sample = tf.to_float(self.neg_vis_probs > tf.random_uniform(tf.shape(self.neg_vis_probs), 0, 1))
            neg_hid_probs = tf.nn.sigmoid(tf.add(tf.matmul(self.neg_vis_sample, self.weights), self.hid_biases))
            neg_associations = tf.matmul(tf.transpose(self.neg_vis_sample), neg_hid_probs) 
            neg_hid_act = tf.reduce_sum(neg_hid_probs, 0)
            neg_vis_act = tf.reduce_sum(self.neg_vis_sample, 0)
            ### Calculate mean squared error
            self.mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - self.neg_vis_sample), 1))

            if self.rbm_version == 'Bengio':
                print('Using rbm_verison:  Bengio')
                pos_associations = tf.matmul(tf.transpose(self.x), self.pos_hid_sample)
                pos_hid_act = tf.reduce_sum(self.pos_hid_sample, 0)
            else:
                print('Using rbm_verison:  Ruslan_new')

        ### Calculate directions to move weights based on gradient (how to change the weights)
        new_weights_increment = (self.momentum*weights_increment + 
                self.lr*((pos_associations-neg_associations)/sym_batch_size - self.weight_c*self.weights))
        new_vis_bias_increment = (self.momentum*vis_bias_increment + 
                                            (self.lr/sym_batch_size)*(pos_vis_act-neg_vis_act))
        new_hid_bias_increment = (self.momentum*hid_bias_increment + 
                                            (self.lr/sym_batch_size)*(pos_hid_act-neg_hid_act))
        updated_weights_increment = tf.assign(weights_increment, new_weights_increment)
        updated_vis_bias_increment = tf.assign(vis_bias_increment, new_vis_bias_increment)
        updated_hid_bias_increment = tf.assign(hid_bias_increment, new_hid_bias_increment)

        ### Update weights and biases
        self.updates = [self.weights.assign_add(updated_weights_increment), 
                   self.vis_biases.assign_add(updated_vis_bias_increment), 
                   self.hid_biases.assign_add(updated_hid_bias_increment), 
                   self.mse]
        print('Finished Building RBM Graph')

    def train(self):   
        '''Trains the RBM. Prints the training and validation set mean squared
        error (reconstruction error) per example to screen. Also saves these 
        errors to disk based on the 'save_path' if 'save_costs_to_csv' is True.
        '''
        print('Training RBM...')
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
            # change momentum
            if epoch > 4:
                m = self.final_momentum
            else:
                m = self.initial_momentum
            total_mse = 0.
            total_batch = int(self.train_dataset.num_examples/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = self.train_dataset.next_batch(self.batch_size)
                # update weights and get mean squared error
                w, vb, hb, error = self.sess.run(self.updates, feed_dict = {self.x: batch_x, self.momentum: m})
                
                # Collect mean squared error for each batch
                total_mse += error

            # Compute average mse per example for each epoch
            avg_mse = total_mse/total_batch

            # Compute validation set average mse per example for each epoch given current state of weights
            validation_avg_mse = self.mse.eval({self.x: self.validation_dataset.features})

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "train mse=", \
                    "{:.9f}".format(avg_mse), "validation mse=", \
                    "{:.9f}".format(validation_avg_mse))

            #collect costs to save to file
            self.train_cost.append(avg_mse)
            self.validation_cost.append(validation_avg_mse)
        print("Optimization Finished!")
        self.end = time.time()
        print('RBM', self.rbm_hidden_layer, 'ran for', (self.end - self.start)/60.)

        if self.save_costs_to_csv:
            self.save_train_and_validation_cost()

    def save_train_and_validation_cost(self):
        '''Saves average train and validation set costs to .csv, all with unique filenames.'''
        # write validation_cost to its own separate file
        name = 'validation_costs_'
        file_path = self.save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self.validation_cost)
        # all error measures in one file
        df_to_disk = pd.DataFrame([self.train_cost, self.validation_cost],
                                    index=[[self.rbm_hidden_layer,self.learning_rate,
                                            self.training_epochs,self.batch_size], ''])
        df_to_disk['error_type'] = ['train_cost', 'validation_cost']
        # create file name and save as .csv
        name = 'all_costs_'
        file_path = self.save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
        df_to_disk.to_csv(file_path)
        print("Train and validation mse saved in file: %s" % file_path)

    def save_model(self, file_name):
        '''Saves model weights and biases to disk'''
        file_path = self.saver.save(self.sess, self.save_path + file_name)
        print("Model saved in file: %s" % file_path)

    def get_reconstruction_images(self, num_images=10, color='magma'):
        '''dispay 10 images from validation set and their reconstructions'''
        if self.rbm_version == 'Hinton_2006':
            reconstructions = self.neg_vis_probs
        elif self.rbm_version in ['Ruslan_new', 'Bengio']:
            reconstructions = self.neg_vis_sample
        dims = get_image_dims(self.n_input)
        encode_decode = self.sess.run(reconstructions, 
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
    from rbm import RBM
    import tensorflow as tf
    import numpy as np
    from dataset import DataSet
    from tensorflow.examples.tutorials.mnist import input_data
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    train_dataset = DataSet(mnist.train.images, mnist.train.labels)
    validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)

    ### SETUP NEURAL NETWORK HYPERPARAMETERS (see class RBM docstring)
    config = {
        'save_path': '/Users/jdy10/Output/biotensorflow/',
        'rbm_hidden_layer': 512,
        'regularizer': None,
        'learning_rate': 0.05, # Learning rate for weights and biases. default 0.1
        'initial_momentum': 0.5, # default 0.5
        'final_momentum': 0.9, # default 0.9
        'weight_cost': 0.00002, # weight-decay. penalty for large weights. default 0.0002
        'training_epochs': 8,
        'batch_size': 100,
        'display_step': 1,
        'rbm_version': 'Hinton_2006', #default 'Hinton_2006'
        'save_costs_to_csv': True
    }

    ### Buid, train, and save model
    with tf.Session() as sess:
        rbm = RBM(sess, config, train_dataset, validation_dataset)  # init config and build graph
        rbm.train() 
        rbm.save_model('model.ckpt')
        rbm.get_reconstruction_images() 

