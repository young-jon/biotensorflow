# TODO
## Short Term

### general
- [ ] github tutorials
- [ ] tensorflow tutorials
- [ ] read deep learning book
	- most important chapters 
		- [ ] 5, 6, 7, 8, 11
	- if time
		- [ ] 14, 15, 20.1-20.4

### biotensorflow (MOST IMPORTANT)
- [ ] modify dnn.py to accept user supplied data 
	- [ ] create small prototyping dataset (~500 x 150 (samples x features/expression))
	- [ ] describe requirements for data (ie., data format, etc.)
	- [ ] update dnn.py to work with any dataset that meets the formatting requirements above
- [ ] modify dnn.py to save model, hidden layers, train error, test error, etc.
- [ ] curate list of all activation functions, optimizers, cost functions, and regularizers that can be used (and have been tested) with dnn.py
- [ ] better understanding of tensorflow cost function api
- [ ] modify dnn.py to run on gpu
- [ ] add different types of regularization to dnn.py
	- [ ] sparse
	- [ ] dropout
	- [ ] denoising
	- [ ] batch normalization
- [ ] add code to dnn.py to decrease learning rate as training progresses (learning rate adaptation)
- [ ] add momentum to dnn.py
- [ ] testing (given a random seed, tests for making sure weights/outputs are computed as expected)
- [ ] visualization 
	- [ ] as another type of testing to make sure algorithm is working
		- [ ] for autoencoder, images of an input next to an image of the reconstruction
	- [ ] to understand results/effectiveness of training
		- [ ] plot of train and test error as a function of epochs (ie., how train and test error decrease over time)


## After short term tasks completed
- algorithms to write:
	- [ ] deep neural network
	- [ ] deep autoencoder 
	- [ ] stacked RBM pretraining (stacked RBM + deep autoencoder = DBN)
	- [ ] greedy autoencoder pretraining
	- [ ] semi-supervised/transfer learning
	- [ ] CNN
	- [ ] multimodal
	- [ ] generative
	- [ ] other
- [ ] model selection code/training pipeline
- [ ] feature selection 
