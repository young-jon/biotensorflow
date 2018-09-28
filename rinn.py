### THIS CODE IS STILL BUGGY AND NOT COMPLETELY TESTED. Multilabel=True mostly doesn't work. Batchnorm doesn't work, others
### as well.
from __future__ import division, print_function, absolute_import
import time
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
from utils import xavier_init



class RINN(object):
    '''
    A Redundant Input Neural Network (Multilayer Perceptron) implementation using the 
    TensorFlow library. See __main__ for example usage.

    Args:

    config (dict): file of hyperparameters
    train_dataset (DataSet): Training data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector.
    validation_dataset (DataSet): same as train_dataset

    '''

    def __init__(self, sess, config, train_dataset, validation_dataset,
                 pretrain_weights=None, pretrain_biases=None):
        self.sess = sess
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.pretrain_weights = pretrain_weights
        self.pretrain_biases = pretrain_biases
        self.save_path = config['save_path']
        self.hidden_layers = config['hidden_layers']
        self.activation = config['activation']
        self.cost_function = config['cost_function']
        self.optimizer = config['optimizer']
        self.regularizer = config['regularizer']
        self.initializer = config['initializer']
        self.learning_rate = config['learning_rate']
        self.training_epochs = config['training_epochs']
        self.batch_size = config['batch_size']
        self.display_step = config['display_step']
        self.save_costs_to_csv = config['save_costs_to_csv']
        self.early_stop_epoch_threshold = config['early_stop_epoch_threshold']

        if self.save_costs_to_csv:
            self._complete_save_path = self.save_path + 'out_' + time.strftime("%m%d%Y_%H:%M:%S") + '/'
            if not os.path.exists(self._complete_save_path):
                os.makedirs(self._complete_save_path)

        self._build_graph()

    def _build_graph(self):
        '''Builds the RINN graph. This function is intended to be called by __init__'''
        ### builds a symbolic deep neural network graph based on the config hyperparameters
        print('Building Graph...')
        self.n_input = self.train_dataset.features.shape[1] # determine from train dataset
        self.n_classes = self.train_dataset.labels.shape[1] # to_one_hot = True

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Store layer weights & biases (initialized using random_normal)
        all_layers = [self.n_input] + self.hidden_layers + [self.n_classes]
        print('Network Architecture: ', all_layers)
        self.weights=[]
        self.biases=[]
        if self.pretrain_weights and self.pretrain_biases:
            print('Using pretrained weights and biases.')
            for i in range(len(self.pretrain_weights)):
                self.weights.append(tf.Variable(self.pretrain_weights[i]))
                self.biases.append(tf.Variable(self.pretrain_biases[i]))
            self.weights.append(tf.Variable(tf.random_normal([all_layers[i+1], all_layers[i+2]])))
            self.biases.append(tf.Variable(tf.random_normal([all_layers[i+2]])))
        elif self.regularizer[0] == tf.contrib.layers.l2_regularizer or self.regularizer[0] == tf.contrib.layers.l1_regularizer:

            ### updated 9_18_18 by removing bias and adding tf.contrib initializers ###
            if self.initializer == 'xavier_custom':
                w1 = tf.get_variable(name='w1', initializer=xavier_init([all_layers[0],all_layers[1]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b1 = tf.get_variable(name='b1', initializer=xavier_init([all_layers[1]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w2 = tf.get_variable(name='w2', initializer=xavier_init([all_layers[0]+all_layers[1],all_layers[2]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b2 = tf.get_variable(name='b2', initializer=xavier_init([all_layers[2]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w3 = tf.get_variable(name='w3', initializer=xavier_init([all_layers[0]+all_layers[2],all_layers[3]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b3 = tf.get_variable(name='b3', initializer=xavier_init([all_layers[3]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w4 = tf.get_variable(name='w4', initializer=xavier_init([all_layers[0]+all_layers[3],all_layers[4]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b4 = tf.get_variable(name='b4', initializer=xavier_init([all_layers[4]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w5 = tf.get_variable(name='w5', initializer=xavier_init([all_layers[0]+all_layers[4],all_layers[5]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b5 = tf.get_variable(name='b5', initializer=xavier_init([all_layers[5]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w6 = tf.get_variable(name='w6', initializer=xavier_init([all_layers[0]+all_layers[5],all_layers[6]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b6 = tf.get_variable(name='b6', initializer=xavier_init([all_layers[6]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w7 = tf.get_variable(name='w7', initializer=xavier_init([all_layers[0]+all_layers[6],all_layers[7]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b7 = tf.get_variable(name='b7', initializer=xavier_init([all_layers[7]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w8 = tf.get_variable(name='w8', initializer=xavier_init([all_layers[0]+all_layers[7],all_layers[8]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b8 = tf.get_variable(name='b8', initializer=xavier_init([all_layers[8]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w9 = tf.get_variable(name='w9', initializer=xavier_init([all_layers[8],all_layers[9]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b9 = tf.get_variable(name='b9', initializer=xavier_init([all_layers[9]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
            else:
                print("Using tf.contrib initializers.")
                w1 = tf.get_variable(name='w1', shape=[all_layers[0],all_layers[1]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b1 = tf.get_variable(name='b1', shape=[all_layers[1]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w2 = tf.get_variable(name='w2', shape=[all_layers[0]+all_layers[1],all_layers[2]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b2 = tf.get_variable(name='b2', shape=[all_layers[2]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w3 = tf.get_variable(name='w3', shape=[all_layers[0]+all_layers[2],all_layers[3]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b3 = tf.get_variable(name='b3', shape=[all_layers[3]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w4 = tf.get_variable(name='w4', shape=[all_layers[0]+all_layers[3],all_layers[4]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b4 = tf.get_variable(name='b4', shape=[all_layers[4]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w5 = tf.get_variable(name='w5', shape=[all_layers[0]+all_layers[4],all_layers[5]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b5 = tf.get_variable(name='b5', shape=[all_layers[5]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w6 = tf.get_variable(name='w6', shape=[all_layers[0]+all_layers[5],all_layers[6]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b6 = tf.get_variable(name='b6', shape=[all_layers[6]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w7 = tf.get_variable(name='w7', shape=[all_layers[0]+all_layers[6],all_layers[7]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b7 = tf.get_variable(name='b7', shape=[all_layers[7]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w8 = tf.get_variable(name='w8', shape=[all_layers[0]+all_layers[7],all_layers[8]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b8 = tf.get_variable(name='b8', shape=[all_layers[8]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w9 = tf.get_variable(name='w9', shape=[all_layers[8],all_layers[9]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b9 = tf.get_variable(name='b9', shape=[all_layers[9]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
            
            self.weights = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
            self.biases = [b1,b2,b3,b4,b5,b6,b7,b8,b9]
            
        else:
            for i in range(len(all_layers)-1):
                self.weights.append(tf.Variable(xavier_init([all_layers[i], all_layers[i+1]])))
                self.biases.append(tf.Variable(tf.random_normal([all_layers[i+1]])))

        # CREATE MODEL
        # create hidden layer 1

        if self.regularizer[0] == tf.nn.dropout:  ### UNTESTED. DON'T USE
            self.model = []
            self.model.append(self.regularizer[0](self.activation(tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0])),self.regularizer[1][0]))
            # create remaining hidden layers
            for j in range(len(self.hidden_layers))[1:]:
                self.model.append(self.regularizer[0](self.activation(tf.add(tf.matmul(self.model[j-1], self.weights[j]), self.biases[j])),self.regularizer[1][j]))
            #create output layer
            self.model.append(tf.add(tf.matmul(self.model[-1], self.weights[-1]), self.biases[-1]))


        elif self.regularizer[0] == tf.nn.batch_normalization: ### DOESN'T WORK YET
            self.model = []
            self.model.append(self.activation(tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0])))
            # create remaining hidden layers
            for j in range(len(self.hidden_layers))[1:]:
                z = tf.add(tf.matmul(self.model[j-1], self.weights[j]), self.biases[j])
                batch_mean, batch_var = tf.nn.moments(z,[0])
                scale = tf.Variable(tf.ones([self.hidden_layers[j]]))
                beta = tf.Variable(tf.zeros([self.hidden_layers[j]]))
                BN = self.regularizer[0](z,batch_mean,batch_var,beta,scale,self.regularizer[1])
                self.model.append(self.activation(BN))
            #create output layer
            self.model.append(tf.add(tf.matmul(self.model[-1], self.weights[-1]), self.biases[-1]))

        else:
            self.model = []
            self.model.append(self.activation(tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0])))
            # create remaining hidden layers
            for j in range(len(self.hidden_layers))[1:]:
                ### new code 5_8_17 ###
                self.model.append(self.activation(tf.add(tf.matmul(tf.concat([self.model[j-1],self.x],1), self.weights[j]), self.biases[j])))
                ### end new code
            #create output layer
            self.model.append(tf.add(tf.matmul(self.model[-1], self.weights[-1]), self.biases[-1]))
            # self.model.append(tf.add(tf.matmul(self.model[-1], tf.nn.sigmoid(self.weights[-1])), self.biases[-1]))
            

 
        # Construct model
        self.logits = self.model[-1]  ### output layer logits

        ### NOTES ### 
        # the output of tf.nn.softmax_cross_entropy_with_logits(logits, y) is an array of the 
        # size of the minibatch (256). each entry in the array is the cross-entropy (scalar value) 
        # for the corresponding image. tf.reduce_mean calculates the mean of this array. Therefore, 
        # the cost variable below (and the cost calculated by sess.run is a scalar value), i.e., the 
        # average cost for a minibatch). see tf.nn.softmax_cross_entropy_with_logits??

        # Define cost (objective function) and optimizer
        self.cost = tf.reduce_mean(self.cost_function(logits = self.logits, labels = self.y)) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.train_step = self.optimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.error = tf.reduce_mean(self.cost_function(logits = self.logits, labels = self.y))
        print('Finished Building DNN Graph')

    def train(self):
        print('Training DNN...')
        # initialize containers for writing results to file
        self.train_cost = []; self.validation_cost = []; 

        # 'Saver' op to save and restore all the variables
        # see https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
        self.saver = tf.train.Saver()

        # Launch the graph
        # with tf.Session() as self.sess:
        # self.sess.run(init)
        self.sess.run(tf.global_variables_initializer())

        # Training cycle
        ### new 1_29_18 for early stopping, updated 9_20_18
        improvement_threshold = 0.99995
        best_validation_cost = np.inf
        best_validation_cost_epoch = 0
        best_validation_error = np.inf
        best_validation_error_epoch = 0
        count = 0
        early_stop = False
        ###
        for epoch in range(self.training_epochs):
            total_cost = 0.
            total_batch = int(self.train_dataset.num_examples/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = self.train_dataset.next_batch(self.batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.train_step, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})
        
                # Collect cost for each batch
                total_cost += c

            # Compute average loss for each epoch
            avg_cost = total_cost/total_batch

            #compute validation set average cost for each epoch given current state of weights
            validation_avg_cost = self.cost.eval({self.x: self.validation_dataset.features, 
                                                    self.y: self.validation_dataset.labels})

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "train cost=", \
                    "{:.9f}".format(avg_cost), "validation cost=", \
                    "{:.9f}".format(validation_avg_cost))

            #collect costs to save to file
            self.train_cost.append(avg_cost)
            self.validation_cost.append(validation_avg_cost)
            
            ### new 1_29_18. updated 9_20_18 ###
            if validation_avg_cost < best_validation_cost * improvement_threshold:
                best_validation_cost = validation_avg_cost
                best_validation_cost_epoch = epoch
                count = 0
                validation_set_error = self.error.eval({self.x: self.validation_dataset.features, 
                                                self.y: self.validation_dataset.labels},
                                                session = self.sess)
                if validation_set_error < best_validation_error:
                    best_validation_error = validation_set_error
                    best_validation_error_epoch = epoch
            else:
                count += 1
            if count == self.early_stop_epoch_threshold:
                print("Stopped due to early stopping!!!")
                early_stop = True
                break
            ### end new
                
        print("Optimization Finished!")

        if self.save_costs_to_csv:
            self.save_train_and_validation_cost()
            
        ### new 1_29_18. updated 9_20_18
        print('Best Validation Set Error: ', best_validation_error)
        return(best_validation_cost, best_validation_error, best_validation_cost_epoch, best_validation_error_epoch, 
            early_stop, epoch)
        ### end new

    def save_train_and_validation_cost(self):
        '''Saves average train and validation set costs to .csv, all with unique filenames'''
        # write validation_cost to its own separate file
        name = 'validation_costs_'
        file_path = self._complete_save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
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
        file_path = self._complete_save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
        df_to_disk.to_csv(file_path)
        print("Train and validation costs saved in file: %s" % file_path)

    ### ADDED 9_20_18###
    def get_validation_cost_and_accuracy(self, validation_dataset, output_layer_activation=tf.identity, multilabel=False):
        '''
        Calculate cost and accuracy for a validation dataset, given dnn weights at time of calling this function.

        ARGS:
        validation_dataset (DataSet): Validating data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector if binary.

        RETURNS:
        Validation set accuracy and average validation set cost over entire validation dataset
        '''
        #calculate validation sets cost

        validation_set_cost = self.cost.eval({self.x: validation_dataset.features, 
                                        self.y: validation_dataset.labels}, 
                                        session = self.sess)

        print('Final validation set cost: ', validation_set_cost)
        #calculate accuracy
        if multilabel:
            # correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(self.logits)), self.y)
            correct_prediction = tf.equal(tf.round(output_layer_activation(self.logits)), tf.round(self.y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            validation_set_error = self.error.eval({self.x: validation_dataset.features, 
                                                self.y: validation_dataset.labels},
                                                session = self.sess)
            validation_set_accuracy = accuracy.eval({self.x: validation_dataset.features, 
                                                self.y: validation_dataset.labels},
                                                session = self.sess)


        else:
            print('ERROR! multilabel must be True.')

        print("Final validation set error:", validation_set_error)
        print("Final validation set accuracy:", validation_set_accuracy)
        return(validation_set_cost, validation_set_error, validation_set_accuracy)

    def get_test_cost_and_accuracy(self, test_dataset, output_layer_activation=tf.identity, multilabel=False):
        '''
        Calculate cost and accuracy for a test dataset, given dnn weights at time of calling this function.

        ARGS:
        test_dataset (DataSet): Testing data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector.

        RETURNS:
        Test set accuracy and average test set cost over entire test dataset
        '''
        #calculate test and validation sets cost

        ### UPDATED 9_18_18 to add validation set cost and accuracy###
        test_set_cost = self.cost.eval({self.x: test_dataset.features, 
                                        self.y: test_dataset.labels}, 
                                        session = self.sess)

        print('Final test set cost: ', test_set_cost)
        #calculate accuracy
        if multilabel:
            # correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(self.logits)), self.y)
            correct_prediction = tf.equal(tf.round(output_layer_activation(self.logits)), tf.round(self.y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_set_accuracy = accuracy.eval({self.x: test_dataset.features, 
                                                self.y: test_dataset.labels},
                                                session = self.sess)
            test_set_error = self.error.eval({self.x: test_dataset.features, 
                                                self.y: test_dataset.labels},
                                                session = self.sess)

        else:
            print('ERROR! multilabel must be True. This branch is untested!')
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_set_accuracy = accuracy.eval({self.x: test_dataset.features, 
                                                self.y: test_dataset.labels},
                                                session = self.sess)
        print("Final test set error:", test_set_error)
        print("Final test set accuracy:", test_set_accuracy)
        return(test_set_cost, test_set_error, test_set_accuracy)

    def save_model(self, file_name):
        ### SAVE MODEL WEIGHTS TO DISK
        file_path = self.saver.save(self.sess, self._complete_save_path + file_name)
        print("Model saved in file: %s" % file_path)



if __name__ == '__main__':
    ### RUN
    import time
    import tensorflow as tf
    import numpy as np
    from utils import read_data_file, sep_data_train_test_val, get_image_dims, xavier_init, rmse, get_sparsity

    ### SET SEEDS
    tf.set_random_seed(18)
    np.random.seed(18)

    ### CSV files should have no headers or text--just a matrix of numbers with rows as instances and columns as features
    f = read_data_file('/home/me/Data/bryan_sim_9_5_18/deg_bin_5000_77.csv')  ###CHANGE 
    g = read_data_file('/home/me/Data/bryan_sim_9_5_18/sga_bin_5000_77.csv')  ###CHANGE

    data = sep_data_train_test_val(g,0.8,0.15,0.05,f)
    train_dataset = data['train']
    test_dataset = data['test']
    validation_dataset = data['validation']


    ### SETUP NEURAL NETWORK HYPERPARAMETERS
    tf.set_random_seed(47456182)
    np.random.seed(47456182)
    ### Hidden layer size
    l = 8

    config = {
        'save_path': '/home/me/Output/res/9_5_18/',
        'hidden_layers': [l,l,l,l,l,l,l,l],
        'activation': tf.nn.relu,
        'cost_function': rmse,
        'optimizer': tf.train.AdamOptimizer,
        'regularizer': [tf.contrib.layers.l1_regularizer, 0.0001],
        'initializer': 'xavier_custom',
        'learning_rate': 0.0003,
        'training_epochs': 3000,
        'batch_size': 128,
        'display_step': 10,
        'early_stop_epoch_threshold': 50,
        'save_costs_to_csv': True
    }

    start = time.time()

    ### Buid, train, and save model
    sess = tf.InteractiveSession()
    rinn = RINN(sess, config, train_dataset, validation_dataset)  # init config and build graph
    best_val_cost, best_val_error, b_c_epoch, b_e_epoch, early, final_epoch = rinn.train()
    f_c, f_e, f_a = rinn.get_validation_cost_and_accuracy(validation_dataset, multilabel=True) 
    #evaluate model on a test set
    c, a, m = rinn.get_test_cost_and_accuracy(test_dataset, multilabel=True, output_layer_activation=tf.identity)
    ae, sp = get_sparsity(rinn.weights, 0.1)

    end = time.time()
    print(end - start)

    ### Successful running with the following package versions from command-line: Segmentation fault when running with IPython
    #tensorflow                1.0.1
    #python                    2.7.14
    #numpy                     1.14.0
