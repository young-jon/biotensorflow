# TODO
## Short Term

### general
- [x] github tutorials
- [ ] tensorflow tutorials
- [ ] read deep learning book
	- most important chapters (assuming you've already read chapters 1-4)
		- [ ] 5, 6, 7, 8, 11
	- if time
		- [ ] 14, 15, 20.1-20.4

### biotensorflow (MOST IMPORTANT)
- [ ] **1.** *(@lujia)*  user supplied data
	- [ ] **1a.** create small *binary* valued prototyping dataset (~5000 x 500 (samples x features)). Ideally from one of the projects in the lab. for example, gene expression dataset.
	- [ ] **1b.** create small *real* valued prototyping dataset similar to above. Ideally from one of the projects in the lab.
	- [ ] **1c.** describe requirements for data (ie., data format, etc.). add this to the function in 1d.
	- [ ] **1d.** write function to create train, validation, and test sets from a single dataset and function to utils.py
- [x] **2.** modify dnn.py to save results
	- [x] **2a.** model 
	- [x] **2b.** hidden layers
	- [x] **2b.** train error, test error, etc.
- [x] **3.** curate list of all activation functions, optimizers, cost functions, and regularizers that can be used (and have been tested) with dnn.py
- [x] **4.** better understanding of tensorflow cost function api
- [ ] **5.** (*@mike*) run dnn.py on gpu
	- [x] **5a.** run dnn.py on single gpu
	- [ ] **5b.** run dnn.py on multiple gpus
- [ ] **6.** (@lujia) add different types of regularization to dnn.py
	- [ ] **6a.** sparse
	- [ ] **6b.** dropout
	- [ ] **6c.** denoising
	- [ ] **6d.** batch normalization
- [ ] **7.** (@mike) add code to dnn.py to decrease learning rate as training progresses (learning rate adaptation)
- [ ] **8.** (@mike) add momentum to dnn.py
- [ ] **9.** (@mike) testing (given a random seed, tests for making sure weights/outputs are computed as expected)
	- [x] **9a.** figure out how to set random seed
	- [ ] **9b.** create testing functions
- [ ] **10.** visualization 
	- [ ] **10a.** as another type of testing to make sure algorithm is working
		- eg., for autoencoder, images of an input next to an image of the reconstruction
	- [ ] **10b.** to understand results/effectiveness of training
		- eg., plot of train and test error as a function of epochs (ie., how train and test error decrease over time)
- [ ] **11.** TensorBoard


## After short term tasks completed
- algorithms to write:
	- [x] deep neural network
	- [x] deep autoencoder 
	- [ ] stacked RBM pretraining (stacked RBM + deep autoencoder = DBN) *@jon*
	- [x] greedy autoencoder pretraining *@jon*
	- [ ] semi-supervised/transfer learning
	- [x] CNN
	- [ ] multimodal
	- [ ] generative
	- [ ] other
- [ ] model selection code/training pipeline
- [ ] feature selection 
- [ ] which high-level api: Keras, TFLearn, TF-Slim, tf.contrib.learn?
