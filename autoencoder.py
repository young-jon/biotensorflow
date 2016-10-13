'''
A Deep Autoencoder implementation using the TensorFlow library. This 
implementation uses the MNIST dataset and will need to be modified for other 
datasets.
'''

# TODO:  better docstring---description of parameters
# TODO:  maybe abstract code to an Autoencoder class...maybe
# TODO:  regularization
# TODO:  testing


import time
import csv
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from __future__ import division, print_function, absolute_import

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


### SETUP NEURAL NETWORK HYPERPARAMETERS
output_folder_path = "/Users/jon/Output/biotensorflow/"
data=mnist
encoder_hidden_layers=[512,256]
activation=tf.nn.sigmoid
cost_function=tf.nn.sigmoid_cross_entropy_with_logits
optimizer=tf.train.RMSPropOptimizer
regularizer=None 
learning_rate=0.01
training_epochs=10
batch_size=256 
display_step=1
examples_to_show=10
output_layer_activation=tf.nn.sigmoid  ### for generating images of the 
# reconstructions only---should match activation used in cost_function. use 
# tf.identity to output affine transformation without an activation function. 

    
n_input = 784 # todo:  calculute this from input x

# tf Graph input
x = tf.placeholder("float", [None, n_input])

# create a list of the sizes of all hidden layers (encoder and decoder)
hidden_layers = encoder_hidden_layers[:]
for h in reversed(encoder_hidden_layers[:-1]):
    hidden_layers.append(h)

# Store layer weights & biases (initialized using random_normal)
all_layers = [n_input] + hidden_layers + [n_input]
weights=[]
biases=[]
for i in range(len(all_layers)-1):
    weights.append(tf.Variable(tf.random_normal([all_layers[i], all_layers[i+1]])))
    biases.append(tf.Variable(tf.random_normal([all_layers[i+1]])))

# CREATE MODEL
# create hidden layer 1
model = []
model.append(activation(tf.add(tf.matmul(x, weights[0]), biases[0])))
# create remaining hidden layers
for j in range(len(hidden_layers))[1:]:
    model.append(activation(tf.add(tf.matmul(model[j-1], weights[j]), biases[j])))
#create output layer
model.append(tf.add(tf.matmul(model[-1], weights[-1]), biases[-1])) 

# Construct model
reconstruction_logits = model[-1]   #output layer i.e., reconstructions

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
cost = tf.reduce_mean(cost_function(reconstruction_logits, x))
# MSE (below) seems to give worse results than cross entropy
# cost = tf.reduce_mean(tf.pow(x - tf.nn.sigmoid(reconstruction_logits), 2))

train_step = optimizer(learning_rate=learning_rate).minimize(cost)

# initialize containers for writing results to file
train_cost = []; test_cost = []; 

# Initializing the variables
init = tf.initialize_all_variables()

# 'Saver' op to save and restore all the variables
# see https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html
# https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        total_cost = 0.
        total_batch = int(data.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = data.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_step, cost], feed_dict={x: batch_x})

            # Collect cost for each batch
            total_cost += c

        # Compute average loss for each epoch
        avg_cost = total_cost/total_batch

        # Compute test set average cost for each epoch given current state of weights
        test_avg_cost = cost.eval({x: data.test.images})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                "cost=", "{:.9f}".format(avg_cost), 
                "test cost=", "{:.9f}".format(test_avg_cost))

        #collect costs to save to file
        train_cost.append(avg_cost)
        test_cost.append(test_avg_cost)

    print("Optimization Finished!")

    ### SAVE .csv files of costs
    ### write test_cost to its own separate file
    name='test_cost_'
    file_path = output_folder_path + name + time.strftime("%m%d%Y_%H;%M") + '.csv'
    with open(file_path, 'a') as f:
        writer=csv.writer(f)
        writer.writerow(test_cost)
    ### all error measures in one file
    df_to_disk = pd.DataFrame([train_cost, test_cost],
                                index=[[hidden_layers,learning_rate,training_epochs,batch_size], ''])
    df_to_disk['error_type'] = ['train_cost', 'test_cost']
    ### create file name and save as .csv
    name = 'all_errors_'
    file_path = output_folder_path + name + time.strftime("%m%d%Y_%H;%M") + '.csv'
    df_to_disk.to_csv(file_path)

    ### SAVE MODEL WEIGHTS TO DISK
    save_path = saver.save(sess, output_folder_path + 'model.ckpt')
    print("Model saved in file: %s" % save_path)

    ### GENERATE FIGURE OF RECONSTRUCTIONS
    encode_decode = sess.run(
        output_layer_activation(reconstruction_logits), 
        feed_dict={x: mnist.test.images[:examples_to_show]}
        )
    ### Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()


    ### See dnn.py for how to load saved model in an interactive session






