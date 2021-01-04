# Fashion-MNIST-GAN
Example of a simple GAN model using keras.

The generator is trained against the discriminator using the script `train_generator.py`.
The generator learns to create 28x28 images of clothes and shoes similar to the ones in the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

A saved generator model can be tested using the script `test_generator.py`. It will output a matrix of mxn results obtained feeding the generator network with mxn random vectors obtained from a normal distribution.

![Examples][./example.png]
