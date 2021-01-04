import tensorflow as tf
import numpy as np
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPooling2D, Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def show_plot(examples, m, n):
    for i in range(m * n):
        pyplot.subplot(m, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    pyplot.figure(figsize=(40,20))
    pyplot.show()

#Load model
model = load_model('generator.h5')
# m is the number of rows, n is the number of columns
m = 4
n = 6
# generate images
latent_points = generate_latent_points(100, m*n)
# generate images
X = model.predict(latent_points)

show_plot(X, m, n)