# TODO
## Short Term

### general
- [ ] github tutorials
- [ ] tensorflow tutorials
- [ ] read deep learning book
	- most important chapters (assuming you've already read chapters 1-4)
		- [ ] 5, 6, 7, 8, 11
	- if time
		- [ ] 14, 15, 20.1-20.4

### biotensorflow (MOST IMPORTANT)
- [ ] **1.** modify dnn.py to accept user supplied data 
	- [ ] **1a.** create small *binary* valued prototyping dataset (~500 x 150 (samples x features/expression))
	- [ ] **1b.** create small *real* valued prototyping dataset similar to above
	- [ ] **1c.** describe requirements for data (ie., data format, etc.)
	- [ ] **1d.** update dnn.py to work with any dataset that meets the formatting requirements above
- [ ] **2.** modify dnn.py to save results
	- [x] **2a.** model 
	- [ ] **2b.** hidden layers
	- [ ] **2b.** train error, test error, etc.
- [ ] **3.** curate list of all activation functions, optimizers, cost functions, and regularizers that can be used (and have been tested) with dnn.py
- [ ] **4.** better understanding of tensorflow cost function api
- [ ] **5.** run dnn.py on gpu
- [ ] **6.** add different types of regularization to dnn.py
	- [ ] **6a.** sparse
	- [ ] **6b.** dropout
	- [ ] **6c.** denoising
	- [ ] **6d.** batch normalization
- [ ] **7.** add code to dnn.py to decrease learning rate as training progresses (learning rate adaptation)
- [ ] **8.** add momentum to dnn.py
- [ ] **9.** testing (given a random seed, tests for making sure weights/outputs are computed as expected)
- [ ] **10.** visualization 
	- [ ] **10a.** as another type of testing to make sure algorithm is working
		- eg., for autoencoder, images of an input next to an image of the reconstruction
	- [ ] **10b.** to understand results/effectiveness of training
		- eg., plot of train and test error as a function of epochs (ie., how train and test error decrease over time)


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
