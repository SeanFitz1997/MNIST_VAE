import tensorflow as tf
import numpy as np


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.inference_net = self.create_inference_net()
        self.generative_net = self.create_generative_net()

    def create_inference_net(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))

        model.add(tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
        assert model.output_shape == (None, 13, 13, 32)

        model.add(tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu'))
        assert model.output_shape == (None, 6, 6, 64)

        model.add(tf.keras.layers.Flatten())
        # No activation
        model.add(tf.keras.layers.Dense(self.latent_dim * 2))

        return model

    def create_generative_net(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)))

        model.add(tf.keras.layers.Dense(units=7*7*32, activation='relu'))
        model.add(tf.keras.layers.Reshape(target_shape=(7, 7, 32)))
        assert model.output_shape == (None, 7, 7, 32)

        model.add(tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
        assert model.output_shape == (None, 14, 14, 64)

        model.add(tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'))
        assert model.output_shape == (None, 28, 28, 32)

        # No activation
        model.add(tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME"))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(
            x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
