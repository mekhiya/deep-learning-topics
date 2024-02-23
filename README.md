# deep-learning-topics
Deep Learning Topics 

**What are Discriminative Models, give examples?**
More like a classification model. Convolutional Neural Network (CNN) is good example. CNN learns features on images from training dataset. FIgures out atterns & decision boundary that distinguishes different class. Later enables multiclass image classification.

**What is the difference between generative and discriminative models?**

![Discriminative-Models-vs-Generative-Models](https://github.com/mekhiya/deep-learning-topics/assets/8952786/544d2b62-eb76-4269-a63e-302c526e35ec)

**What are Autoencoders and How Do They Work?**
Autoencoders are special neural networks that learn how to recreate the information. They can -  reduce the number of features in a dataset, extracting meaningful features from data, detecting anomalies, and generating new data. It is trained to copy its input to its output.
For eg:- given an image of a handwritten digit, an autoencoder first encodes the image into a lower dimensional latent representation, then decodes the latent representation back to an image
Stable Diffusion uses latent diffusion model.
![Autoencoders-1](https://github.com/mekhiya/deep-learning-topics/assets/8952786/4488be3e-4723-4e63-b8cf-d99c481e53d5)

**What are some popular autoencoders, mention few?**
Unlike the AR language model(which uses auto-regression to find next word), BERT is autoencoder(AE) language model. It reconstruct the original data from corrupted input. corrupted input = [MASK] . Mask replaces the original token into in the pre-train phase.
![Autoencoders](https://github.com/mekhiya/deep-learning-topics/assets/8952786/3f715dca-4156-4f67-8570-23e4d478063c)

**What is the role of the Loss function in Autoencoders, & how is it different from other NN?**
Loss function in Autoencoders vs loss function in NN

How do autoencoders differ from (PCA)?

Which one is better for reconstruction linear autoencoder or PCA?

How can you recreate PCA with neural networks?

Can You Explain How Autoencoders Can be Used for Anomaly Detection?

What are some applications of AutoEncoders

How can uncertainty be introduced into Autoencoders, & what are the benefits and challenges of doing so?

Can you explain what VAE is and describe its training process?

Explain what Kullback-Leibler (KL) divergence is & why does it matter in VAEs?

Can you explain what reconstruction loss is & it’s function in VAEs?

What is ELBO & What is this trade-off between reconstructionQuality & regularization?

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
