import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from progress.bar import Bar
from vae import (VAE, log_normal_pdf,
                 compute_loss, train_step)
from config import NTH_SAVE, GPU


''' Load and Preporcess Data '''
# Load dataset
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

# Reshape data and encode data
train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(
    test_images.shape[0], 28, 28, 1).astype('float32')
train_images /= 255.
test_images /= 255.

# Make binary values
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

TRAIN_BUFFER = len(train_images)
TEST_BUFFER = len(test_images)
BATCH_SIZE = 100
BUFFER_SIZE = len(train_images)

# Batch and Shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(TRAIN_BUFFER).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    test_images).shuffle(TEST_BUFFER).batch(BATCH_SIZE)

''' Create Model & Optimizer '''
latent_dim = 50
model = VAE(latent_dim)
optimizer = tf.keras.optimizers.Adam(1e-4)


''' Create Save Checkpoints '''
checkpoint_dir = './training_checkpoints'
training_img_dir = 'training_images'
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

# Load models from last checkpoint
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
num_prev_epochs = len([name for name in os.listdir(training_img_dir)])


''' Define Training Loop '''
epochs = 100
num_examples_to_generate = 16
number_of_batches = int(TRAIN_BUFFER / BATCH_SIZE)
seed = tf.random.normal([num_examples_to_generate, latent_dim])


def generate_and_save_images(model, epoch, test_input, show=False):
    predictions = model.sample(test_input)
    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.show() if show else plt.savefig(
        '{}/image_at_epoch_{:04d}.png'.format(training_img_dir, epoch))


def train(train_dataset, test_dataset, epochs):
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        num_batches = int(BUFFER_SIZE / BATCH_SIZE)
        with Bar('Epoch {}'.format(epoch), max=num_batches) as bar: 
            for image_batch in train_dataset:
                train_step(model, image_batch, optimizer)
                bar.next()

        end_time = time.time()

        # Save the model every N epoch
        checkpoint.save(file_prefix=checkpoint_prefix)

        # Epoch logging
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))

        print('Epoch:{}, loss: {}, time {} sec'.format(
            epoch, loss.result(), end_time - start_time))

        generate_and_save_images(model, epoch, seed)


try:
    with tf.device(GPU):
        train(train_dataset, test_dataset, epochs)
except:
    train(train_dataset, test_dataset, epochs)
