### RANDOM CODE


### run the following to reload a checkpoint file in a new ipython session. 
### config and datasets must be defined identically to when checkpoint was trained.
if __name__ == '__main__':
    ### Restore graph and model from checkpoint
    from dnn_class import DNN
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
        'regularizer': None,
        'learning_rate': 0.001,
        'training_epochs': 3,
        'batch_size': 100,
        'display_step': 1
    }

    sess = tf.InteractiveSession()
    dnn = DNN(sess, config, train_dataset, validation_dataset)
    saver = tf.train.Saver()
    save_path = config['save_path']
    saver.restore(sess, save_path+'model.ckpt') ### restores previous session with stored state of all variables
    print("Model restored from file: %s" % save_path)

    ### example calculations after model is restored
    dnn.weights[0].eval()
    dnn.cost.eval({dnn.x: validation_dataset.features, dnn.y: validation_dataset.labels})
    dnn.model[0].eval({dnn.x: train_dataset.features})
    dnn.cost.eval({dnn.x: test_dataset.features, dnn.y: test_dataset.labels})
    dnn.cost.eval({dnn.x: train_dataset.features, dnn.y: train_dataset.labels})
    # get hidden layer values using test set 
	h1 = model[0].eval({x: mnist.test.images})
	h2 = model[1].eval({x: mnist.test.images})
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


### to get a tensorboard graph
# run code below after sess.run(init)
summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
# then open new terminal window and navigate to save location of log_simple_graph. activate tensorflow and run 
# the following command at the command line
tensorboard --logdir=log_simple_graph

### print very long and hard to read version of graph in ipython
#add print statement below after 'init = tf.initialize_all_variables()''
print(tf.get_default_graph().as_graph_def())


### to set all random seeds, add the following to the top of a script, after imports
tf.reset_default_graph()
np.random.seed(2)
tf.set_random_seed(2)



### INTERACTIVE SESSION FOR LOADING SAVED MODEL AND CALCULATING HIDDEN LAYER VALUES
#load saved model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, '/Users/jon/Output/biotensorflow/model.ckpt') ### restores previous session with stored state of all variables
sess.close()

#other
save_path = '/Users/jon/Output/biotensorflow/'
sess = tf.InteractiveSession()
ckpt = tf.train.get_checkpoint_state(save_path)
saver = tf.train.Saver()
saver.restore(sess, save_path + 'model.ckpt') ### restores previous session with stored state of all variables
print("Model restored from file: %s" % file_path)

import tensorflow as tf
saver = tf.train.import_meta_graph("model.ckpt.meta")
sess = tf.Session()
saver.restore(sess, "model.ckpt")











### NOTES ###

### NOTES ABOUT RELU UNITS AND AUTOENCODERS (see relu_results.xlsx)
# I've noticed that when using relu units with dnn.py, setting the hidden layers to be really large (2000)
# leads to a drastic increase in the average value of the output layer logits, leading to a drastic increase 
# in the validation set cost. you see this increase even when the test set accuracy improves. i.e., when i 
# train with a dnn with hidden layers = [100,50] and relu, my validation set cost is very low but my accuracy
# is also very low. however, when i train with hidden_layers = [2000,1000] and relu, my validation set cost is 
# very high, but my accuracy significantly improves. when using larger layers with relu, the average value of 
# the output logits is also significantly increased relative to smaller sized layers with relu. THEREFORE: 
# maybe shouldn't be using relu units when depending on the cross-entropy cost to determine best model!!! It 
# seems that softmax_cross_entropy_with_logits in the tensorflow library is slightly different than if i were 
# calculating this same value by hand. the tensorflow version is optimized for stability and to avoid overflow. I 
# assume that it is similar to the sigmoid_cross_entropy_with_logits:
# max(logits, 0) - logits * targets + log(1 + exp(-abs(logits))),
# where they use the logits in the actual cross_entropy without applying the activation function; therefore, the
# magnitude of the logits affects the values of the cross-entropy. One way to deal with this is to simply divide
# the average train, validation, and test set cross-entropies by np.mean(np.abs(logits)), np.std(logits), or 
# np.median(np.abs(logits)) after the script has terminated. 