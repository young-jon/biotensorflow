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

tf.reset_default_graph()
np.random.seed(2)
tf.set_random_seed(2)

# TODO:  change variable name to previous code
# TODO:  implement pseudolikelihood (see scikit, deep learning tutorials)
class RBM(object):
    '''
    A Restricted Boltzmann Machine implementation using the TensorFlow library. 
    See __main__ for example usage. 
    Eventually, there will be three versions of this code based on popular 
    implementations. They will be called 'Hinton_2006', 'Ruslan_new',  and 
    'Bengio', and you will be able to use them with an argument in config.
    Currently, 'Hinton_2006' is implemented and is based on code from Hinton and 
    Salakhutdinov's 2006 Science paper 
    (http://www.cs.toronto.edu/~hinton/code/rbm.m). 'Ruslan_new' is based on 
    code from http://www.cs.toronto.edu/~rsalakhu/code_DBM/rbm.m. 'Bengio' is 
    based on pseudocode from page 33 of 
    http://www.iro.umontreal.ca/~bengioy/papers/ftml.pdf

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

        self._build_graph()

    def _build_graph(self):  ### Different from DNN
        '''Builds the RBM graph. This function is intended to be called by __init__'''
        ### builds a symbolic deep autoencoder graph based on the config hyperparameters
        self.start=time.time()
        print('Building Graph...')
        self.n_input = self.train_dataset.features.shape[1] # determine from train dataset

        # Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.sym_batch_size = tf.to_float(tf.shape(self.x)[0])  ### symbolic, can pass in data with diff # examples
        print('RBM Network Architecture: ', [self.n_input,self.rbm_hidden_layer,self.n_input])

        # Graph Hyperparameters
        self.lr = tf.constant(self.learning_rate, dtype=tf.float32)
        self.weight_c = tf.constant(self.weight_cost, dtype=tf.float32)
        self.momentum = tf.placeholder(tf.float32)

        # Graph Variables
        self.weights = tf.Variable(tf.random_normal([self.n_input, self.rbm_hidden_layer])) 
        self.hid_biases = tf.Variable(tf.random_normal([self.rbm_hidden_layer]))
        self.vis_biases = tf.Variable(tf.random_normal([self.n_input]))
        ### hinton
        # weights = tf.Variable(tf.random_normal([n_input, rbm_hidden_layer], stddev=0.1)) #remove .1 to use mean 0 dev 1
        # could also use stddev = 0.01 or 0.001
        # hid_biases = tf.Variable(tf.zeros([rbm_hidden_layer]))
        # vis_biases = tf.Variable(tf.zeros([n_input]))

        self.weights_increment  = tf.Variable(tf.zeros([self.n_input, self.rbm_hidden_layer]))
        self.vis_bias_increment = tf.Variable(tf.zeros([self.n_input]))
        self.hid_bias_increment = tf.Variable(tf.zeros([self.rbm_hidden_layer]))

        ### BUILD MODEL
        ### Start positive phase of 1-step Contrastive Divergence
        self.pos_hid_probs = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.weights), self.hid_biases))
        self.pos_associations = tf.matmul(tf.transpose(self.x), self.pos_hid_probs)
        self.pos_hid_act = tf.reduce_sum(self.pos_hid_probs, 0)
        self.pos_vis_act = tf.reduce_sum(self.x, 0)

        ### Start negative phase
        ### get a sample of the hidden unit states from the distribution created from pos_hid_probs
        self.pos_hid_sample = self.pos_hid_probs > tf.random_uniform(tf.shape(self.pos_hid_probs), 0, 1)
        self.pos_hid_sample = tf.to_float(self.pos_hid_sample)

        self.neg_vis_probs=tf.nn.sigmoid(tf.add(tf.matmul(self.pos_hid_sample,tf.transpose(self.weights)),self.vis_biases))
        self.neg_hid_probs = tf.nn.sigmoid(tf.add(tf.matmul(self.neg_vis_probs, self.weights), self.hid_biases))
        self.neg_associations = tf.matmul(tf.transpose(self.neg_vis_probs), self.neg_hid_probs) 
        self.neg_hid_act = tf.reduce_sum(self.neg_hid_probs, 0)
        self.neg_vis_act = tf.reduce_sum(self.neg_vis_probs, 0)

        ### Calculate directions to move weights based on gradient (how to change the weights)
        self.new_weights_increment = (self.momentum*self.weights_increment + 
                self.lr*((self.pos_associations-self.neg_associations)/self.sym_batch_size - self.weight_c*self.weights))
        self.new_vis_bias_increment = (self.momentum*self.vis_bias_increment + 
                                            (self.lr/self.sym_batch_size)*(self.pos_vis_act-self.neg_vis_act))
        self.new_hid_bias_increment = (self.momentum*self.hid_bias_increment + 
                                            (self.lr/self.sym_batch_size)*(self.pos_hid_act-self.neg_hid_act))
        self.updated_weights_increment = tf.assign(self.weights_increment, self.new_weights_increment)
        self.updated_vis_bias_increment = tf.assign(self.vis_bias_increment, self.new_vis_bias_increment)
        self.updated_hid_bias_increment = tf.assign(self.hid_bias_increment, self.new_hid_bias_increment)

        ### Calculate mean squared error
        self.mse = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - self.neg_vis_probs), 1))

        ### Update weights and biases
        self.updates = [self.weights.assign_add(self.updated_weights_increment), 
                   self.vis_biases.assign_add(self.updated_vis_bias_increment), 
                   self.hid_biases.assign_add(self.updated_hid_bias_increment), self.mse]
        print('Finished Building RBM Graph')

    def train(self):   
        print('Training RBM...')
        # initialize containers for writing results to file
        self.train_error = []; self.validation_error = []; 

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
            self.train_error.append(avg_mse)
            self.validation_error.append(validation_avg_mse)
        print("Optimization Finished!")
        self.end=time.time()
        print('ran for', (self.end - self.start)/60.)

        # if self.save_costs_to_csv:
        #     self.save_train_and_validation_cost()

    def get_reconstruction_images(self):
        '''dispay 10 images from validation set and their reconstructions'''
        encode_decode = self.sess.run(self.neg_vis_probs, feed_dict={self.x: self.validation_dataset.features[:10]})
        ### Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(self.validation_dataset.features[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()


#     ### SAVE .csv files of error measures
#     # write validation_cost to its own separate file
#     name='validation_costs_'
#     file_path = save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
#     with open(file_path, 'a') as f:
#         writer=csv.writer(f)
#         writer.writerow(validation_error)
#     # all error measures in one file
#     df_to_disk = pd.DataFrame([train_error, validation_error],
#                                 index=[[rbm_hidden_layer,learning_rate,training_epochs,batch_size], ''])
#     df_to_disk['error_type'] = ['train_error', 'validation_error']
#     # create file name and save as .csv
#     name = 'all_costs_'
#     file_path = save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
#     df_to_disk.to_csv(file_path)
#     print("Train and validation errors saved in file: %s" % file_path)

#     ### SAVE MODEL WEIGHTS TO DISK
#     file_path = saver.save(sess, save_path + save_model_filename)
#     print("Model saved in file: %s" % file_path)

#     if get_reconstruction_images:
#         '''dispay 10 images from validation set and their reconstructions'''
#         encode_decode = sess.run(neg_vis_probs, feed_dict={x: validation_dataset.features[:10]})
#         ### Compare original images with their reconstructions
#         f, a = plt.subplots(2, 10, figsize=(10, 2))
#         for i in range(10):
#             a[0][i].imshow(np.reshape(validation_dataset.features[i], (28, 28)))
#             a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
#         f.show()
#         plt.draw()
#         plt.waitforbuttonpress()


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

    ### SETUP NEURAL NETWORK HYPERPARAMETERS
    config = {
        'save_path': '/Users/jon/Output/biotensorflow/',
        'rbm_hidden_layer': 512,
        'regularizer': None,
        'learning_rate': 0.05, # Learning rate for weights and biases. default 0.1
        'initial_momentum': 0.5, # default 0.5
        'final_momentum': 0.9, # default 0.9
        'weight_cost': 0.00002, # weight-decay. penalty for large weights. default 0.0002
        'training_epochs': 3,
        'batch_size': 100,
        'display_step': 1,
        'save_costs_to_csv': True
    }


    ### Buid, train, and save model
    with tf.Session() as sess:
        rbm = RBM(sess, config, train_dataset, validation_dataset)  # init config and build graph
        rbm.train() 
        # rbm.save_model('model.ckpt')
        rbm.get_reconstruction_images() # argument here is the output layer activation
        # function for generating images of the reconstructions. This should match the activation 
        # used in the cost_function. Use tf.identity to output affine transformation without an 
        # activation function. 

