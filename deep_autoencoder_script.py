from __future__ import division, print_function, absolute_import
import time
import csv
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from dataset import DataSet

'''
A Deep Autoencoder implementation using the TensorFlow library.
'''

# TODO:  better docstring---description of parameters
# TODO:  regularization
# TODO:  testing


# Get and define data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_dataset = DataSet(mnist.train.images, mnist.train.labels)
validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)

### SETUP NEURAL NETWORK HYPERPARAMETERS
save_path = "/Users/jon/Output/biotensorflow/"
save_model_filename = 'model.ckpt'
encoder_hidden_layers=[256, 128]
activation=tf.nn.sigmoid
cost_function=tf.nn.sigmoid_cross_entropy_with_logits
optimizer=tf.train.RMSPropOptimizer
regularizer=None 
learning_rate=0.01
training_epochs=20
batch_size=256 
display_step=1
get_reconstruction_images=True
output_layer_activation=tf.nn.sigmoid  ### for generating images of the 
# reconstructions only---should match activation used in cost_function. use 
# tf.identity to output affine transformation without an activation function. 

print('Building Graph...')
n_input = train_dataset.features.shape[1] # determine from train dataset

# tf Graph input
x = tf.placeholder("float", [None, n_input])

# create a list of the sizes of all hidden layers (encoder and decoder)
hidden_layers = encoder_hidden_layers[:]
for h in reversed(encoder_hidden_layers[:-1]):
    hidden_layers.append(h)

# Store layer weights & biases (initialized using random_normal)
all_layers = [n_input] + hidden_layers + [n_input]
print('Network Architecture: ', all_layers)
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
logits = model[-1]   #output layer i.e., reconstructions

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
cost = tf.reduce_mean(cost_function(logits, x))
# MSE (below) seems to give worse results than cross entropy
# cost = tf.reduce_mean(tf.pow(x - tf.nn.sigmoid(logits), 2))

train_step = optimizer(learning_rate=learning_rate).minimize(cost)
print('Finished Building DA Graph')

# initialize containers for writing results to file
train_cost = []; validation_cost = []; 

print('Training DA...')
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
        total_batch = int(train_dataset.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train_dataset.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_step, cost], feed_dict={x: batch_x})

            # Collect cost for each batch
            total_cost += c

        # Compute average loss for each epoch
        avg_cost = total_cost/total_batch

        # Compute test set average cost for each epoch given current state of weights
        validation_avg_cost = cost.eval({x: validation_dataset.features})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                "train cost=", "{:.9f}".format(avg_cost), 
                "validation cost=", "{:.9f}".format(validation_avg_cost))

        #collect costs to save to file
        train_cost.append(avg_cost)
        validation_cost.append(validation_avg_cost)

    print("Optimization Finished!")

    ### SAVE .csv files of error measures
    # write validation_cost to its own separate file
    name='validation_costs_'
    file_path = save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
    with open(file_path, 'a') as f:
        writer=csv.writer(f)
        writer.writerow(validation_cost)
    # all error measures in one file
    df_to_disk = pd.DataFrame([train_cost, validation_cost],
                                index=[[hidden_layers,learning_rate,training_epochs,batch_size], ''])
    df_to_disk['error_type'] = ['train_cost', 'validation_cost']
    # create file name and save as .csv
    name = 'all_costs_'
    file_path = save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
    df_to_disk.to_csv(file_path)
    print("Train and validation costs saved in file: %s" % file_path)

    ### SAVE MODEL WEIGHTS TO DISK
    file_path = saver.save(sess, save_path + save_model_filename)
    print("Model saved in file: %s" % file_path)

    ### GENERATE FIGURE OF RECONSTRUCTIONS
    if get_reconstruction_images:
        '''dispay 10 images from validation set and their reconstructions'''
        encode_decode = sess.run(output_layer_activation(logits), 
                            feed_dict={x: validation_dataset.features[:10]})
        ### Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(validation_dataset.features[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()







