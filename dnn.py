'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''


def dnn(data,hidden_layers,activation,optimizer,regularizer,learning_rate,training_epochs,batch_size,display_step):
    

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


    # Create model
    model = []
    model.append(activation(tf.add(tf.matmul(x, weights[0]), biases[0])))
    for j in range(len(hidden_layers))[1:]:
        model.append(activation(tf.add(tf.matmul(model[j-1], weights[j]), biases[j])))
    model.append(tf.add(tf.matmul(model[j], weights[j+1]), biases[j+1]))  #output layer


    # Construct model
    pred = model[-1]

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    train_step = optimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(data.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = data.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_step, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: data.test.images, y: data.test.labels}))


### MAIN
### RUN DNN
import tensorflow as tf
from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


dnn(data=mnist, 
    hidden_layers=[200,100], 
    activation=tf.nn.relu, 
    optimizer=tf.train.AdamOptimizer, 
    regularizer=0, 
    learning_rate=0.001, 
    training_epochs=50, 
    batch_size=100, 
    display_step=1)




