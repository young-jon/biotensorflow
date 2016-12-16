# TODO:  modify DNN to initialize to pretrain weights
# TODO:  test save_model and restoring model after saving multiple different architectures in same directory
# TODO:  other types of weight intiialization (see deep learning tutorials)
# TODO:  in save_model, also need to save config and paths to file
# TODO:  better docstring---description of parameters
# TODO:  maybe abstract code to a DNN class
# TODO:  regularization
# TODO:  testing

from __future__ import division, print_function, absolute_import
import time
import csv
import pandas as pd
import tensorflow as tf
import numpy as np


class DNN(object):
    '''
    A Deep Neural Network (Multilayer Perceptron) implementation using the 
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
        self.learning_rate = config['learning_rate']
        self.training_epochs = config['training_epochs']
        self.batch_size = config['batch_size']
        self.display_step = config['display_step']
        self.save_costs_to_csv = config['save_costs_to_csv']

        self._build_graph()

    def _build_graph(self):
        '''Builds the DNN graph. This function is intended to be called by __init__'''
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
        else:
            for i in range(len(all_layers)-1):
                self.weights.append(tf.Variable(tf.random_normal([all_layers[i], all_layers[i+1]])))
                self.biases.append(tf.Variable(tf.random_normal([all_layers[i+1]])))

        # CREATE MODEL
        # create hidden layer 1

        if self.regularizer[0] == tf.nn.dropout:
            self.model = []
            self.model.append(self.regularizer[0](self.activation(tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0])),self.regularizer[1][0]))
            # create remaining hidden layers
            for j in range(len(self.hidden_layers))[1:]:
                self.model.append(self.regularizer[0](self.activation(tf.add(tf.matmul(self.model[j-1], self.weights[j]), self.biases[j])),self.regularizer[1][j]))
            #create output layer
            self.model.append(tf.add(tf.matmul(self.model[-1], self.weights[-1]), self.biases[-1]))


        elif self.regularizer[0] == tf.nn.batch_normalization:
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
                self.model.append(self.activation(tf.add(tf.matmul(self.model[j-1], self.weights[j]), self.biases[j])))
            #create output layer
            self.model.append(tf.add(tf.matmul(self.model[-1], self.weights[-1]), self.biases[-1]))
            

 
        # Construct model
        self.logits = self.model[-1]  ### output layer logits

        ### NOTES ### 
        # the output of tf.nn.softmax_cross_entropy_with_logits(logits, y) is an array of the 
        # size of the minibatch (256). each entry in the array is the cross-entropy (scalar value) 
        # for the corresponding image. tf.reduce_mean calculates the mean of this array. Therefore, 
        # the cost variable below (and the cost calculated by sess.run is a scalar value), i.e., the 
        # average cost for a minibatch). see tf.nn.softmax_cross_entropy_with_logits??

        # Define cost (objective function) and optimizer
        self.cost = tf.reduce_mean(self.cost_function(self.logits, self.y))
        self.train_step = self.optimizer(learning_rate=self.learning_rate).minimize(self.cost)
        print('Finished Building DNN Graph')

    def train(self):
        print('Training DNN...')
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
        print("Optimization Finished!")

        if self.save_costs_to_csv:
            self.save_train_and_validation_cost()

    def save_train_and_validation_cost(self):
        '''Saves average train and validation set costs to .csv, all with unique filenames'''
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

    def get_test_cost_and_accuracy(self, test_dataset):
        '''
        Calculate cost and accuracy for a test dataset.

        ARGS:
        test_dataset (DataSet): Testing data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector.

        RETURNS:
        Test set accuracy and average test set cost over entire test dataset
        '''
        #calculate test cost
        test_set_cost = self.cost.eval({self.x: test_dataset.features, 
                                        self.y: test_dataset.labels}, 
                                        session = self.sess)
        print('Test set cost: ', test_set_cost)
        #calculate accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_set_accuracy = accuracy.eval({self.x: test_dataset.features, 
                                            self.y: test_dataset.labels},
                                            session = self.sess)
        print("Accuracy:", test_set_accuracy)
        return(test_set_cost, test_set_accuracy)

    def save_model(self, file_name):
        ### SAVE MODEL WEIGHTS TO DISK
        file_path = self.saver.save(self.sess, self.save_path + file_name)
        print("Model saved in file: %s" % file_path)

    def test_model(self, train_dataset, validation_dataset, test_dataset): 
        ### TESTS TO MAKE SURE THAT COSTS ARE COMPUTED AS EXPECTED USING CURRENT TRAIN_VALIDATION DATASET
        print("Testing model with default parameters!")
        ### DEFAULT NEURAL NETWORK HYPERPARAMETERS
        config = {
            'save_path': '/Users/jon/Output/biotensorflow/',
            'hidden_layers': [150,60],
            'activation': tf.nn.relu,
            'cost_function': tf.nn.softmax_cross_entropy_with_logits,
            'optimizer': tf.train.AdamOptimizer,
            #'regularizer': [None],
            'regularizer':[tf.nn.batch_normalization,1e-5],#batch_normalization_epsilon
            #'regularizer': [tf.nn.dropout,[0.6,1]],#dropout_rate
            'learning_rate': 0.001,
            'training_epochs': 3,
            'batch_size': 100,
            'display_step': 4, #training_epochs + 1, don't want to display anything
            'save_costs_to_csv': False
        }

        ### GENERATE RANDOM SEEDS
        tensorflow_seed = np.array([0, 42, 1234, 1776, 1729])
        numpy_seed = np.array([1729, 1776, 1234, 42, 0])

        test_costs = [None] * 5 

        for count in range(0,5): 
            #print(count)
            tf.set_random_seed(tensorflow_seed[count])
            np.random.seed(numpy_seed[count])

            with tf.Session() as sess: 
                #use subprocesses to silent 

                dnn = DNN(sess, config, train_dataset, validation_dataset)
                dnn.train()

                #evaluate model on a test set
                c, a = dnn.get_test_cost_and_accuracy(test_dataset)
                test_costs[count] = c

        print("#####")
        print("Average cost", sum(test_costs)/5)


### EXAMPLE USAGE
if __name__ == '__main__':
    from dnn import DNN
    import tensorflow as tf
    import numpy as np
    from dataset import DataSet
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    train_dataset = DataSet(mnist.train.images, mnist.train.labels)
    validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)
    test_dataset = DataSet(mnist.test.images, mnist.test.labels)

    ### SETUP NEURAL NETWORK HYPERPARAMETERS
    config = {
        'save_path': '/Users/jon/Output/biotensorflow/',
        'hidden_layers': [100,50],
        'activation': tf.nn.relu,
        'cost_function': tf.nn.softmax_cross_entropy_with_logits,
        'optimizer': tf.train.AdamOptimizer,
        #'regularizer': [None],
        'regularizer':[tf.nn.batch_normalization,1e-5],#batch_normalization_epsilon
        #'regularizer': [tf.nn.dropout,[0.7,1]],#dropout_rate
        'learning_rate': 0.001,
        'training_epochs': 3,
        'batch_size': 100,
        'display_step': 1,
        'save_costs_to_csv': True
    }

    ### Buid, train, and save model
    with tf.Session() as sess:
        dnn = DNN(sess, config, train_dataset, validation_dataset)  # init config and build graph
        dnn.train() 
        dnn.save_model('model.ckpt')
        #evaluate model on a test set
        c, a = dnn.get_test_cost_and_accuracy(test_dataset)

    ### test_model 
    dnn.test_model(train_dataset, validation_dataset, test_dataset)    

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
