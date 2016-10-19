from __future__ import division, print_function, absolute_import
import time
import csv
import pandas as pd
import tensorflow as tf
from dataset import DataSet
from tensorflow.examples.tutorials.mnist import input_data


'''
A Deep Neural Network (Multilayer Perceptron) implementation using the 
TensorFlow library. This implementation currently uses the MNIST dataset 
and will need to be modified for other datasets.
'''

# TODO:  better docstring---description of parameters
# TODO:  maybe abstract code to a DNN class
# TODO:  regularization
# TODO:  testing


# Get and define data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_dataset = DataSet(mnist.train.images, mnist.train.labels)
validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)
test_dataset = DataSet(mnist.test.images, mnist.test.labels)

### SETUP NEURAL NETWORK HYPERPARAMETERS
save_path = "/Users/jon/Output/biotensorflow/"
save_model_filename = 'model.ckpt'
hidden_layers=[100,50]
activation=tf.nn.relu
cost_function=tf.nn.softmax_cross_entropy_with_logits #function with params = logits, y
optimizer=tf.train.AdamOptimizer
regularizer=None 
learning_rate=0.001
training_epochs=3
batch_size=100 
display_step=1

print('Building Graph...')
n_input = train_dataset.features.shape[1] # determine from train dataset
n_classes = train_dataset.labels.shape[1]  

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layer weights & biases (initialized using random_normal)
all_layers = [n_input] + hidden_layers + [n_classes]
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
logits = model[-1]  ### output layer logits

### NOTES ### 
# the output of tf.nn.softmax_cross_entropy_with_logits(logits, y) is an array of the 
# size of the minibatch (256). each entry in the array is the cross-entropy (scalar value) 
# for the corresponding image. tf.reduce_mean calculates the mean of this array. Therefore, 
# the cost variable below (and the cost calculated by sess.run is a scalar value), i.e., the 
# average cost for a minibatch). see tf.nn.softmax_cross_entropy_with_logits??

# Define cost (objective function) and optimizer
cost = tf.reduce_mean(cost_function(logits, y))
train_step = optimizer(learning_rate=learning_rate).minimize(cost)
print('Finished Building DNN Graph')

# initialize containers for writing results to file
train_cost = []; validation_cost = []; 

print('Training DNN...')
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
            _, c = sess.run([train_step, cost], feed_dict={x: batch_x, y: batch_y})
    
            # Collect cost for each batch
            total_cost += c

        # Compute average loss for each epoch
        avg_cost = total_cost/total_batch

        #compute validation set average cost for each epoch given current state of weights
        validation_avg_cost = cost.eval({x: validation_dataset.features, 
                                        y: validation_dataset.labels})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "train cost=", \
                "{:.9f}".format(avg_cost), "validation cost=", \
                "{:.9f}".format(validation_avg_cost))

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

    ### TEST MODEL
    #calculate test cost
    test_set_cost = cost.eval({x: test_dataset.features, 
                                y: test_dataset.labels})
    print('Test set cost:', test_set_cost)
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_dataset.features, y: test_dataset.labels}))

    ### SAVE MODEL WEIGHTS TO DISK
    file_path = saver.save(sess, save_path + save_model_filename)
    print("Model saved in file: %s" % file_path)


### To reload model in same ipython session with same graph defined, run:
# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
# saver.restore(sess, save_path + save_model_filename)

### To reload model in new ipython session see random.py

### example usage after model loaded 
# weights[0].eval()  # print 1st layer weights
# h1=model[0].eval({x: train_dataset})  # get hidden layer 1 values using train data
### save hidden layer values
# import numpy as np
# np.savetxt(save_path + 'h1_train.csv', h1, delimiter=",")  ### np.loadtxt('h1_train.csv', delimiter=",")








