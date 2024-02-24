# deep-learning-topics
Deep Learning Topics 

**What are Discriminative Models?**

More like a classification model. Convolutional Neural Network (CNN) is good example. CNN learns features on images from training dataset. FIgures out atterns & decision boundary that distinguishes different class. Later enables multiclass image classification.

**generative vs discriminative models**

![Discriminative-Models-vs-Generative-Models](https://github.com/mekhiya/deep-learning-topics/assets/8952786/544d2b62-eb76-4269-a63e-302c526e35ec)

**What are Autoencoders**

Autoencoders are special neural networks that learn how to recreate the information. They can -  reduce the number of features in a dataset, extracting meaningful features from data, detecting anomalies, and generating new data. It is trained to copy its input to its output.
For eg:- given an image of a handwritten digit, an autoencoder first encodes the image into a lower dimensional latent representation, then decodes the latent representation back to an image
Stable Diffusion uses latent diffusion model.
![Autoencoders-1](https://github.com/mekhiya/deep-learning-topics/assets/8952786/4488be3e-4723-4e63-b8cf-d99c481e53d5)

**popular autoencoders**

Unlike the AR language model(which uses auto-regression to find next word), BERT is autoencoder(AE) language model. It reconstruct the original data from corrupted input. corrupted input = [MASK] . Mask replaces the original token into in the pre-train phase.
![Autoencoders](https://github.com/mekhiya/deep-learning-topics/assets/8952786/3f715dca-4156-4f67-8570-23e4d478063c)

**role of the Loss function in Autoencoders vs role of Loss function in other NN**

_Loss function in Autoencoders_
loss function for autoencoders are reconstruction loss. It measure the difference between the model input and output. The reconstruction error is calculated using Mean Squared Error (MSE) Loss & L1 Loss popular option.
Autoencoder is trained for data compression. Loss function in Autoencoder measures result vs reality of encoding-decoding process, process of reconstructing compressed input.

_Loss function in other NN_
loss function measures success of NN model in performing a certain task - regression or classification (binary, multiclass). Loss is minimised with backpropagation step.

**Liner vs non-linear models**

Linear model result graph is has a constant rate of change, plotted with a straight line as the dependent variable changes in response to the independent variable. 
Non-linear model result graph does not have a constant rate of change.
linear equation gives straight line, whereas non-linear equation gives conic curve - circle, parabola etc
Linear model - linear regression model used for predicting the price of a house by analyzing historical data. Linear Regression can be of 2 types - Simple and Multiple Linear Regression.
Non- linear models - Neural Networks

**autoencoders vs PCA**

Principal Component Analysis (PCA) is an unsupervised learning algorithm technique. It examines relations between set of variables. It reduces dimensions in large data sets. It transform large set of variables into a smaller one without lossing most of the information.

Autoencoders used for non linear and complex data.

**recreate PCA with neural networks**

In simple neural network architecture when inout layers have more nodes then next layer. Next layers performs a compression or dimension reduction similar. This is similar ot PCA action.

**Explain How Autoencoders Can be Used for Anomaly Detection?**

Autoencoder is a neural network that is used to learn efficient codings of unlabeled data (unsupervised learning). An autoencoder learns two functions: an encoding function that transforms the input data, and a decoding function that recreates the input data from the encoded representation.
Autoencoder learns patterns , basic representation of normal data and reconstruction with minimum error. Reconstruction error can be used in identifying anomaly from normal data.

**applications of AutoEncoders**

Dimension reduction, Anomaly detection, Denoising, Feature extraction, feature learning, recommendation systems, data generation, convolution autoencoders, generative modeling, image compression, image generation

**uncertainty |  Autoencoders |  challenges**

An autoencoder could misclassify input errors that are different from those in the training set or changes in underlying relationships that a human would notice. Another drawback is you may eliminate the vital information in the input data.

**VAE training process**

Variational autoencoder (VAE) is a technique used to improve the quality of AI generated images you create with the text-to-image model Stable Diffusion. VAE encodes the image into a latent space and then that latent space is decoded into a new, higher quality image.
Variational autoencoders are unsupervised learning methods, don't require labels in addition to the data inputs. 
During the training process, the VAE adjusts the parameters of the encoder and decoder networks to minimize reconstruction error &  KL divergence between the variational distribution and the true posterior distribution.

**Kullback-Leibler (KL) divergence in VAEs**
KL quantifies how much one probability distribution differs from another probability distribution. The KL divergence between two distributions Q and P is often stated using the following notation: KL(P || Q)

**reconstruction loss in VAEs**
Reconstruction loss ensures a close match of output with input in VAEs. 
loss function is used to minimize reconstruction error. It is regularized by the KL divergence between the probability distribution of training data (the prior distribution) and the distribution of latent variables learned by the VAE (the posterior distribution).

**ELBO | reconstructionQuality | regularization**
ELBO -  Evidence Lower Bound loss
Regularisation - techniques that calibrate models to minimize the adjusted loss function and prevent overfitting or underfitting.
The goal of regularization is to encourage models to learn the broader patterns within the data rather than memorizing it.

Can you explain the training & optimization process of VAEs?

How would you balance reconstructionQuality and latent space regularization in a practical Variational Autoencoder implementation?

What is Reparametrization trick and why is it important?

What is DGG "Deep Clustering via a Gaussian-mixture Variational Autoencoder (VAE)” with Graph Embedding

How does a neural network with one layer and one input and output compare to a logistic regression?

In a logistic regression model, will all the gradient descent algorithms lead to the same model if run for a long time?

What is padding and why it’s used in Convolutional Neural Networks (CNNs)?

Padded Convolutions: What are Valid and Same Paddings?

What is stride in CNN and why is it used?

What is the impact of Stride size on CNNs?

What is Pooling, what is the intuition behind it and why is it used in CNNs?

What are common types of pooling in CNN?

Why min pooling is not used?

What is translation invariance and why is it important?

How does a 1D Convolutional Neural Network (CNN) work?

What are Recurrent Neural Networks, and walk me through the architecture of RNNs.

What are the main disadvantages of RNNs, especially in Machine Translation Tasks?

What are some applications of RNN?

What technique is commonly used in RNNs to combat the Vanishing Gradient Problem?

What are LSTMs and their key components?

What limitations of RNN that LSTMs do and don’t address and how?

What is a gated recurrent unit (GRU) and how is it different from LSTMs?

Describe how Generative Adversarial Networks (GANs) work and the roles of the generator and discriminator in learning.

What are token embeddings and what is their function?

What is Multi-Head Self-Attention and how does it enable more effective processing of sequences in Transformers?

What are transformers and why are they important in combating problems of models like RNN and LSTMs?

Walk me through the architecture of transformers.

What are positional encodings and how are they calculated?

Why do we add positional encodings to Transformers but not to RNN or LSTMs?
