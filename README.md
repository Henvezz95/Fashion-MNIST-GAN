# Fashion-MNIST-GAN
Example of a simple GAN model using keras.

The generator is trained against the discriminator using the script `train_generator.py`.
The generator learns to create 28x28 images of clothes and shoes similar to the ones in the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

A saved generator model can be tested using the script `test_generator.py`. It will output a matrix of mxn results obtained feeding the generator network with mxn random vectors obtained from a normal distribution.

Example of a random set of 4x4 images obtained from the network:

![Examples](https://github.com/Henvezz95/Fashion-MNIST-GAN/blob/main/example.png)
