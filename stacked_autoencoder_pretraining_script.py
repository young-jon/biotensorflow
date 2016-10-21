from __future__ import division, print_function, absolute_import
import time
import csv
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataset import DataSet
from tensorflow.examples.tutorials.mnist import input_data

'''
Greedy layer-wise autoencoder pretrainig implementation using the TensorFlow library. This 
implementation uses the MNIST dataset and will need to be modified for other 
datasets.
'''

# TODO:  better docstring---description of parameters
# TODO:  maybe abstract code to an Autoencoder class...maybe
# TODO:  regularization
# TODO:  testing
#1. separate graph for each layer. save previous layer output as numpy.array to pass forward
#2. cmgreen


# Import MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_dataset = DataSet(mnist.train.images, mnist.train.labels)
validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)

### SETUP NEURAL NETWORK HYPERPARAMETERS
output_folder_path = "/Users/jdy10/Output/biotensorflow/"
hidden_layers=[256,128]
activation=tf.nn.sigmoid
cost_function=tf.nn.sigmoid_cross_entropy_with_logits
# optimizer=tf.train.GradientDescentOptimizer
optimizer=tf.train.RMSPropOptimizer
regularizer=None 
learning_rate=0.01
training_epochs=3
batch_size=256 
display_step=1
examples_to_show=10
output_layer_activation=tf.nn.sigmoid  ### for generating images of the 
# reconstructions only---should match activation used in cost_function. use 
# tf.identity to output affine transformation without an activation function. 
ENCODER_WEIGHTS = []
DECODER_WEIGHTS = []
ENCODER_BIASES = []
DECODER_BIASES = []

for layer in hidden_layers:
    print('Pretraining', layer, 'layer...')
    n_input = train_dataset.features.shape[1] # todo:  calculute this from input x

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])

    # Store layer weights & biases (initialized using random_normal)
    w1 = tf.Variable(tf.random_normal([n_input, layer]))
    b1 = tf.Variable(tf.random_normal([layer]))
    w2 = tf.Variable(tf.random_normal([layer, n_input]))
    b2 = tf.Variable(tf.random_normal([n_input]))

    # CREATE MODEL
    model = []
    model.append(activation(tf.add(tf.matmul(x, w1), b1)))
    model.append(tf.add(tf.matmul(model[0], w2), b2))

    # Construct model
    reconstruction_logits = model[-1]

    # Define cost (objective function) and optimizer
    cost = tf.reduce_mean(cost_function(reconstruction_logits, x))
    train_step = optimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

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

            ### compute validation set average cost for each epoch given current state of weights
            validation_avg_cost = cost.eval({x: validation_dataset.features})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), 
                    "cost=", "{:.9f}".format(avg_cost), 
                    "validation cost=", "{:.9f}".format(validation_avg_cost))


        print("Optimization Finished!")

        start_time = time.time()

        ### NEW
        new_x = model[0].eval({x: train_dataset.features})
        new_validation_x = model[0].eval({x: validation_dataset.features})
        ENCODER_WEIGHTS.append(w1.eval())
        DECODER_WEIGHTS.append(w2.eval())
        ENCODER_BIASES.append(b1.eval())
        DECODER_BIASES.append(b2.eval())

	train_dataset = DataSet(new_x, np.round(new_x[:,0]))
	validation_dataset = DataSet(new_validation_x, np.round(new_validation_x[:,0]))

	# end_time = time.time()
	# print('The new code for ' + str(layer) + ' ran for %.2fm' % ((end_time - start_time) / 60.))


### FINETUNING
print('Finetuning...')
train_dataset = DataSet(mnist.train.images, mnist.train.labels)
validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)
n_input = train_dataset.features.shape[1]
PRETRAIN_WEIGHTS = ENCODER_WEIGHTS + list(reversed(DECODER_WEIGHTS))
PRETRAIN_BIASES = ENCODER_BIASES + list(reversed(DECODER_BIASES))

# tf Graph input
x = tf.placeholder("float", [None, n_input])

# create a list of the sizes of all hidden layers (encoder and decoder)
all_hidden_layers = hidden_layers[:]
for h in reversed(hidden_layers[:-1]):
    all_hidden_layers.append(h)

# Store layer weights & biases (initialized using random_normal)
all_layers = [n_input] + all_hidden_layers + [n_input]
weights=[]
biases=[]
for i in range(len(all_layers)-1):
    weights.append(tf.Variable(PRETRAIN_WEIGHTS[i]))
    biases.append(tf.Variable(PRETRAIN_BIASES[i]))

# CREATE MODEL
# create hidden layer 1
model = []
model.append(activation(tf.add(tf.matmul(x, weights[0]), biases[0])))
# create remaining hidden layers
for j in range(len(all_hidden_layers))[1:]:
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
train_cost = []; validation_cost = []; 

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

        #compute validation set average cost for each epoch given current state of weights
        validation_avg_cost = cost.eval({x: validation_dataset.features})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                "cost=", "{:.9f}".format(avg_cost), 
                "validation cost=", "{:.9f}".format(validation_avg_cost))

        #collect costs to save to file
        train_cost.append(avg_cost)
        validation_cost.append(validation_avg_cost)

    print("Optimization Finished!")

    ### GENERATE FIGURE OF RECONSTRUCTIONS
    encode_decode = sess.run(
        output_layer_activation(reconstruction_logits), 
        feed_dict={x: validation_dataset.features[:examples_to_show]}
        )
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(validation_dataset.features[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()








