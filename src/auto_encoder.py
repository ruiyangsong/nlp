import numpy as np
from keras import models, layers
import matplotlib.pyplot as plt
from src.utils import split_data
def auto_encoder(data_pth = '../data/mode_padding.npz', outdir='../fig'):
    data = np.load(data_pth)
    x = data['x']
    y = data['y']
    x_train, y_train, x_test, y_test = split_data(x,y)
    # in order to plot in a 2D figure
    encoding_dim = 2
    input = layers.Input(shape=(x_train.shape[1],))
    # encoder layers
    encoded = layers.Dense(64, activation='relu')(input)
    encoded = layers.Dense(10, activation='relu')(encoded)
    encoder_output = layers.Dense(encoding_dim)(encoded)
    # decoder layers
    decoded = layers.Dense(10, activation='relu')(encoder_output)
    decoded = layers.Dense(64, activation='relu')(decoded)
    decoded = layers.Dense(x_train.shape[1], activation='tanh')(decoded)
    # construct the autoencoder model
    autoencoder = models.Model(input=input, output=decoded)
    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    # training
    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=128,
                    shuffle=True)
    # construct the encoder model for plotting
    encoder = models.Model(input=input, output=encoder_output)
    # plotting
    encoded_imgs = encoder.predict(x_train)
    plt.figure(figsize=(5,5),dpi=200)
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_train)
    plt.colorbar()
    # plt.show()
    plt.savefig('%s/auto-encoder2Dim.png' % outdir)  # pngfile

if __name__ == '__main__':
    auto_encoder()