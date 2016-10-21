from __future__ import division, print_function, absolute_import
import time
import csv
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataset import DataSet
from deep_autoencoder import DA
from tensorflow.examples.tutorials.mnist import input_data

# TODO:  remove dummy labels in SA.train after modify DataSet to accept unsupervised data
# TODO:  fix for copying config in init
# TODO:  better docstring---description of parameters
# TODO:  regularization
# TODO:  testing
#1. separate graph for each layer. save previous layer output as numpy.array to pass forward
#2. cmgreen

class SA(object):
    '''
    Greedy layer-wise autoencoder pretrainig implementation using the TensorFlow library. This 
    implementation uses the MNIST dataset and will need to be modified for other 
    dataset
    '''
    def __init__(self, sess, config, train_dataset, validation_dataset):
        '''Same as DNN except for encoder_hidden_layers'''
        self.sess = sess
        self.config = config.copy()  ### this is a bad fix, but need to modify config for 
        ### each autoencoder and python dictionaries are global...
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
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
        self.encoder_weights = []
        self.decoder_weights = []
        self.encoder_biases = []
        self.decoder_biases = []

    def train(self): 
        '''Builds the SA graph. This function is intended to be called by __init__'''
        self.SA_encoder_hidden_layers = self.encoder_hidden_layers[:]
        autoencoders = []
        for i, layer in enumerate(self.SA_encoder_hidden_layers):
            print('Pretraining', layer, 'layer...')
            self.config['encoder_hidden_layers'] = [layer]
            # Build single hidden layer autoencoder graph
            autoencoders.append(DA(self.sess, self.config, self.train_dataset, self.validation_dataset))
            # Train single hidden layer autoencoder
            autoencoders[i].train()
            # Get autoencoder hidden layer to use as input to next autoencoder
            new_train=autoencoders[i].model[0].eval({autoencoders[i].x: self.train_dataset.features})
            new_validation=autoencoders[i].model[0].eval({autoencoders[i].x: self.validation_dataset.features})
            # Create DataSet objects of new data
            self.train_dataset = DataSet(new_train, np.round(new_train[:,0]))
            self.validation_dataset = DataSet(new_validation, np.round(new_validation[:,0]))
            # Save encoder and decoder weights and biases for single hidden layer autoencoder
            self.encoder_weights.append(autoencoders[i].weights[0].eval())
            self.decoder_weights.append(autoencoders[i].weights[1].eval())
            self.encoder_biases.append(autoencoders[i].biases[0].eval())
            self.decoder_biases.append(autoencoders[i].biases[1].eval())

    def get_pretraining_weights_and_biases(self):
        pretrain_weights = self.encoder_weights + list(reversed(self.decoder_weights))
        pretrain_biases = self.encoder_biases + list(reversed(self.decoder_biases))
        return(pretrain_weights, pretrain_biases)


### EXAMPLE USAGE
if __name__ == '__main__':
    from pretrain import SA
    from dnn import DNN
    from deep_autoencoder import DA
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
        'encoder_hidden_layers': [256,128],
        'activation': tf.nn.sigmoid,
        'cost_function': tf.nn.sigmoid_cross_entropy_with_logits,
        'optimizer': tf.train.RMSPropOptimizer,
        'regularizer': None,
        'learning_rate': 0.01,
        'training_epochs': 5,
        'batch_size': 256,
        'display_step': 1,
        'save_costs_to_csv': True
    }

    ### SA PRETRAINING
    with tf.Session() as sess:
        sa = SA(sess, config, train_dataset, validation_dataset)  # init config and build graph
        sa.train()
        weights, biases = sa.get_pretraining_weights_and_biases()

    ### DA FINETUNING
    ### for code testing
    # train_dataset = DataSet(mnist.train.images, mnist.train.labels)
    # validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)
    print('Finetuning...')
    with tf.Session() as sess:
        da = DA(sess, config, train_dataset, validation_dataset, pretrain_weights=weights, pretrain_biases=biases)
        da.train() 
        da.save_model('model.ckpt')
        da.get_reconstruction_images(tf.nn.sigmoid) # argument here is the output layer activation
        # function for generating images of the reconstructions. This should match the activation 
        # used in the cost_function. Use tf.identity to output affine transformation without an 
        # activation function.  

        ### FOR SUPERVISED DNN FINETUNING USE THE FOLLOWING CODE 
        # config = {
        #     'save_path': '/Users/jon/Output/biotensorflow/',
        #     'hidden_layers': [100,50],
        #     'activation': tf.nn.relu,
        #     'cost_function': tf.nn.softmax_cross_entropy_with_logits,
        #     'optimizer': tf.train.AdamOptimizer,
        #     'regularizer': None,
        #     'learning_rate': 0.001,
        #     'training_epochs': 3,
        #     'batch_size': 100,
        #     'display_step': 1,
        #     'save_costs_to_csv': True
        # }
        # index = int(len(weights)/2)
        # encoder_weights = weights[:index]
        # encoder_biases = biases[:index]
        # dnn = DNN(sess, config, train_dataset, validation_dataset, 
        #             pretrain_weights=encoder_weights, pretrain_biases=encoder_biases)
        # dnn.train() 
        # dnn.save_model('model.ckpt')
        # c, a = dnn.get_test_cost_and_accuracy(validation_dataset)


