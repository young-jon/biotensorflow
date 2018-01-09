# biotensorflow

Code repository with deep learning algorithms developed for biological applications---written using Python 2.7 and TensorFlow and developed by members of the Lu lab. In general, all code should be thoroughly tested before pushing to the repository.

How to run a Stack Restricted Boltzmann Machine - Deep Autoencoder (SRBM-DA).  An SRBM-DA is also known as a Deep Belief Network (DBN):

```python
import tensorflow as tf
import numpy as np
from pretrain import SRBM
from deep_autoencoder import DA
from dataset import DataSet
from tensorflow.examples.tutorials.mnist import input_data

### Setup data. Can use any data defined as a DataSet class
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_dataset = DataSet(mnist.train.images, mnist.train.labels)
validation_dataset = DataSet(mnist.validation.images, mnist.validation.labels)

### SETUP NEURAL NETWORK HYPERPARAMETERS
config = {
    'save_path': '/Users/jon/Output/biotensorflow/',
    'encoder_hidden_layers': [256,128],
    'regularizer': None,
    'learning_rate': 0.05, # Learning rate for weights and biases. default 0.1
    'initial_momentum': 0.5, # default 0.5
    'final_momentum': 0.9, # default 0.9
    'weight_cost': 0.00002, # weight-decay. penalty for large weights. default 0.0002
    'training_epochs': 8,
    'batch_size': 100,
    'display_step': 1,
    'rbm_version': 'Hinton_2006', #default 'Hinton_2006'
    'save_costs_to_csv': True
}

### SRBM PRETRAINING
print('Pretraining...')
with tf.Session() as sess:
    srbm = SRBM(sess, config, train_dataset, validation_dataset)  # init config and build graph
    srbm.train()
    weights, biases = srbm.get_pretraining_weights_and_biases()

### DA FINETUNING
### change/add config hyperparameters
config['activation'] = tf.nn.sigmoid
#tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, tf.nn.elu
config['cost_function'] = tf.nn.sigmoid_cross_entropy_with_logits
config['optimizer'] = tf.train.AdamOptimizer
#AdadeltaOptimizer,AdagradOptimizer,AdamOptimizer,MomentumOptimizer,RMSPropOptimizer,GradientDescentOptimizer
config['learning_rate'] = 0.01

print('Finetuning...')
with tf.Session() as sess:
    da = DA(sess, config, train_dataset, validation_dataset, pretrain_weights=weights, pretrain_biases=biases)
    da.train() 
    da.save_model('model.ckpt')
    da.get_reconstruction_images(tf.nn.sigmoid)
```

For more examples and documentation, see pretrain.py, rbm.py, and deep_autoencoder.py.

### Practical recommendations for training autoencoder with binary gene expression data
-use AdamOptimizer and xavier initialization. 
(remember reconstruction error issues with relu)
