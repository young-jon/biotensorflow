from __future__ import division, print_function, absolute_import
import time
import csv
import pandas as pd
import tensorflow as tf
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


# Get data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


### SETUP NEURAL NETWORK HYPERPARAMETERS
output_folder_path = "/Users/jon/Output/biotensorflow/"
data=mnist
hidden_layers=[200,100]
activation=tf.nn.relu
cost_function=tf.nn.softmax_cross_entropy_with_logits
optimizer=tf.train.AdamOptimizer
regularizer=None 
learning_rate=0.001
training_epochs=10
batch_size=100 
display_step=1

    
n_input = 784 # calculute this from input x
n_classes = 10 # calculate this from y

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layer weights & biases (initialized using random_normal)
all_layers = [n_input] + hidden_layers + [n_classes]
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
            _, c = sess.run([train_step, cost], feed_dict={x: batch_x, y: batch_y})
    
            # Collect cost for each batch
            total_cost += c

        # Compute average loss for each epoch
        avg_cost = total_cost/total_batch

        #compute test set average cost for each epoch given current state of weights
        test_avg_cost = cost.eval({x: data.test.images, y: data.test.labels})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost), "test cost=", \
                "{:.9f}".format(test_avg_cost))

        #collect costs to save to file
        train_cost.append(avg_cost)
        test_cost.append(test_avg_cost)

    print("Optimization Finished!")

    ### SAVE .csv files of error measures
    # write test_cost to its own separate file
    name='test_cost_'
    file_path = output_folder_path + name + time.strftime("%m%d%Y_%H;%M") + '.csv'
    with open(file_path, 'a') as f:
        writer=csv.writer(f)
        writer.writerow(test_cost)
    # all error measures in one file
    df_to_disk = pd.DataFrame([train_cost, test_cost],
                                index=[[hidden_layers,learning_rate,training_epochs,batch_size], ''])
    df_to_disk['error_type'] = ['train_cost', 'test_cost']
    # create file name and save as .csv
    name = 'all_errors_'
    file_path = output_folder_path + name + time.strftime("%m%d%Y_%H;%M") + '.csv'
    df_to_disk.to_csv(file_path)

    ### TEST MODEL
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: data.test.images, y: data.test.labels}))

    ### SAVE MODEL WEIGHTS TO DISK
    save_path = saver.save(sess, output_folder_path + 'model.ckpt')
    print("Model saved in file: %s" % save_path)




### INTERACTIVE SESSION FOR LOADING SAVED MODEL AND CALCULATING HIDDEN LAYER VALUES
#load saved model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, output_folder_path+'model.ckpt') ### restores previous session with stored state of all variables
print("Model restored from file: %s" % output_folder_path)
# print 1st layer weights
w1=weights[0].eval()
print(w1)

# get hidden layer values using test set 
h1 = model[0].eval({x: data.test.images})
h2 = model[1].eval({x: data.test.images})
# save hidden layer values
import numpy as np
np.savetxt(output_folder_path + 'h1_test.csv', h1, delimiter=",")  ### np.loadtxt('foo.csv', delimiter=",")
np.savetxt(output_folder_path + 'h2_test.csv', h2, delimiter=",")  ### np.loadtxt('foo.csv', delimiter=",")

#first row 
h1[0,:]
#first column 
h1[:,0]
#first 3 column values for first 3 rows
h1[0:3,0:3]

sess.close()








