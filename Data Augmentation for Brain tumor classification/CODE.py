# Import Required Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# Path to the dataset
train_dir = 'DATA/Training'
test_dir = 'DATA/Testing'

# Image data generator for preprocessing and loading images
datagen = ImageDataGenerator(rescale=1. / 255,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# Load and preprocess training images from subdirectories
train_data = datagen.flow_from_directory(train_dir,
                                         target_size=(64, 64),
                                         batch_size=32,
                                         class_mode='categorical',
                                         shuffle=True)


# Define the Generator Model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, input_dim=100))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh'))
    return model


# Define the Discriminator Model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# Define the DCGAN Model (Combined Generator and Discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# Instantiate generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator with a lower learning rate
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

# GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)

# Compile the GAN with a lower learning rate
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5), loss='binary_crossentropy')


# Training the DCGAN Model
def train_dcgan(generator, discriminator, gan, epochs, batch_size, train_data):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train the discriminator
        real_images, _ = next(train_data)  # Get a batch of real images
        real_labels = np.ones((half_batch, 1))  # Labels for real images

        noise = np.random.normal(0, 1, (half_batch, 100))  # Generate random noise
        fake_images = generator.predict(noise)  # Generate fake images
        fake_labels = np.zeros((half_batch, 1))  # Labels for fake images

        # Train discriminator on real and fake images
        d_loss_real = discriminator.train_on_batch(real_images[:half_batch], real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))  # Generate noise
        valid_labels = np.ones((batch_size, 1))  # Labels for the generator to fool the discriminator

        g_loss = gan.train_on_batch(noise, valid_labels)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")
            save_generated_images(epoch, generator)


# Function to Save Generated Images
def save_generated_images(epoch, generator, output_dir='Generated_Images'):
    noise = np.random.normal(0, 1, (16, 100))
    generated_images = generator.predict(noise)

    # Save images to file
    for i in range(generated_images.shape[0]):
        img = generated_images[i] * 127.5 + 127.5  # Rescale images to [0, 255]
        img = np.uint8(img)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f'{output_dir}/generated_image_{epoch}_{i}.png')


# Train the Model
epochs = 10000
batch_size = 32
train_dcgan(generator, discriminator, gan, epochs, batch_size, train_data)

# Save the Models
generator.save('models/generator_model.h5')
discriminator.save('models/discriminator_model.h5')

# Load and Compile the Generator Model (for Generating Images)
generator = tf.keras.models.load_model('/models/generator_model.h5')
generator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5))  # Compile to avoid warning


# Generate Images Using the Loaded Model
def generate_images(generator, num_images=10):
    noise = np.random.normal(0, 1, (num_images, 100))
    generated_images = generator.predict(noise)

    for i in range(generated_images.shape[0]):
        img = generated_images[i] * 127.5 + 127.5  # Rescale images to [0, 255]
        img = np.uint8(img)
        plt.imshow(img)
        plt.axis('off')
        plt.show()


generate_images(generator)
