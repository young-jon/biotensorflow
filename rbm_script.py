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

### SETUP NEURAL NETWORK HYPERPARAMETERS
save_path= '/Users/jon/Output/biotensorflow/'
save_model_filename = 'model.ckpt'
rbm_hidden_layer= 512
regularizer= None
learning_rate= 0.05  # Learning rate for weights and biases. default 0.1
training_epochs= 8
batch_size= 100
weight_cost= 0.00002 ### weight-decay. penalty for large weights. default 0.0002
display_step= 1
save_costs_to_csv= True
initial_momentum= 0.5 # default 0.5
final_momentum= 0.9 # default 0.9
get_reconstruction_images=True


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_dataset = DataSet(mnist.train.images, mnist.train.labels)
validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)

start=time.time()
print('Building Graph...')
n_input = train_dataset.features.shape[1] # determine from train dataset

# Graph input
x = tf.placeholder("float", [None, n_input])
sym_batch_size = tf.to_float(tf.shape(x)[0])  ### symbolic, so can pass in datasets with different # examples
print('RBM Network Architecture: ', [n_input,rbm_hidden_layer,n_input])

# Graph Hyperparameters
lr = tf.constant(learning_rate, dtype=tf.float32)
weight_c = tf.constant(weight_cost, dtype=tf.float32)
momentum = tf.placeholder(tf.float32)

# Variables
weights = tf.Variable(tf.random_normal([n_input, rbm_hidden_layer])) 
hid_biases = tf.Variable(tf.random_normal([rbm_hidden_layer]))
vis_biases = tf.Variable(tf.random_normal([n_input]))
### hinton
# weights = tf.Variable(tf.random_normal([n_input, rbm_hidden_layer], stddev=0.1)) #remove .1 to use mean 0 stddev 1
# could also use stddev = 0.01 or 0.001
# hid_biases = tf.Variable(tf.zeros([rbm_hidden_layer]))
# vis_biases = tf.Variable(tf.zeros([n_input]))

weights_increment  = tf.Variable(tf.zeros([n_input, rbm_hidden_layer]))
vis_bias_increment = tf.Variable(tf.zeros([n_input]))
hid_bias_increment = tf.Variable(tf.zeros([rbm_hidden_layer]))

### Start positive phase of 1-step Contrastive Divergence
pos_hid_probs = tf.nn.sigmoid(tf.add(tf.matmul(x, weights), hid_biases))
pos_associations = tf.matmul(tf.transpose(x), pos_hid_probs)
pos_hid_act = tf.reduce_sum(pos_hid_probs, 0)
pos_vis_act = tf.reduce_sum(x, 0)

### Start negative phase
### get a sample of the hidden unit states from the distribution created from pos_hid_probs
pos_hid_sample = pos_hid_probs > tf.random_uniform(tf.shape(pos_hid_probs), 0, 1)
pos_hid_sample = tf.to_float(pos_hid_sample)

neg_vis_probs = tf.nn.sigmoid(tf.add(tf.matmul(pos_hid_sample, tf.transpose(weights)), vis_biases))
neg_hid_probs = tf.nn.sigmoid(tf.add(tf.matmul(neg_vis_probs, weights), hid_biases))
neg_associations = tf.matmul(tf.transpose(neg_vis_probs), neg_hid_probs) 
neg_hid_act = tf.reduce_sum(neg_hid_probs, 0)
neg_vis_act = tf.reduce_sum(neg_vis_probs, 0)

### Calculate directions to move weights based on gradient (how to change the weights)
new_weights_increment = (momentum*weights_increment + 
                            lr*((pos_associations-neg_associations)/sym_batch_size - weight_c*weights))
new_vis_bias_increment = momentum*vis_bias_increment + (lr/sym_batch_size)*(pos_vis_act-neg_vis_act)
new_hid_bias_increment = momentum*hid_bias_increment + (lr/sym_batch_size)*(pos_hid_act-neg_hid_act)
updated_weights_increment = tf.assign(weights_increment, new_weights_increment)
updated_vis_bias_increment = tf.assign(vis_bias_increment, new_vis_bias_increment)
updated_hid_bias_increment = tf.assign(hid_bias_increment, new_hid_bias_increment)

### Calculate mean squared error
mse = tf.reduce_mean(tf.reduce_sum(tf.square(x - neg_vis_probs), 1))

### Update weights and biases
updates = [weights.assign_add(updated_weights_increment), 
           vis_biases.assign_add(updated_vis_bias_increment), 
           hid_biases.assign_add(updated_hid_bias_increment), mse]
print('Finished Building RBM Graph')

# initialize containers for writing results to file
train_error = []; validation_error = []; 

print('Training RBM...')
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
        # change momentum
        if epoch > 4:
            m = final_momentum
        else:
            m = initial_momentum
        total_mse = 0.
        total_batch = int(train_dataset.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train_dataset.next_batch(batch_size)
            # update weights and get mean squared error
            w, vb, hb, error = sess.run(updates, feed_dict = {x: batch_x, momentum: m})

            # Collect mean squared error for each batch
            total_mse += error

        # Compute average mse per example for each epoch
        avg_mse = total_mse/total_batch

        # Compute validation set average mse per example for each epoch given current state of weights
        validation_avg_mse = mse.eval({x: validation_dataset.features})

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                "train mse=", "{:.9f}".format(avg_mse), 
                "validation mse=", "{:.9f}".format(validation_avg_mse))

        #collect costs to save to file
        train_error.append(avg_mse)
        validation_error.append(validation_avg_mse)

    print("Optimization Finished!")
    end=time.time()
    print('ran for', (end - start)/60.)


    ### SAVE .csv files of error measures
    # write validation_cost to its own separate file
    name='validation_costs_'
    file_path = save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
    with open(file_path, 'a') as f:
        writer=csv.writer(f)
        writer.writerow(validation_error)
    # all error measures in one file
    df_to_disk = pd.DataFrame([train_error, validation_error],
                                index=[[rbm_hidden_layer,learning_rate,training_epochs,batch_size], ''])
    df_to_disk['error_type'] = ['train_error', 'validation_error']
    # create file name and save as .csv
    name = 'all_costs_'
    file_path = save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
    df_to_disk.to_csv(file_path)
    print("Train and validation errors saved in file: %s" % file_path)

    ### SAVE MODEL WEIGHTS TO DISK
    file_path = saver.save(sess, save_path + save_model_filename)
    print("Model saved in file: %s" % file_path)

    if get_reconstruction_images:
        '''dispay 10 images from validation set and their reconstructions'''
        encode_decode = sess.run(neg_vis_probs, feed_dict={x: validation_dataset.features[:10]})
        ### Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(validation_dataset.features[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()

