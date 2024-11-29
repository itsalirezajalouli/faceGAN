# Imports
import os
import gdown
import keras
from keras import ops
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from zipfile import ZipFile

# Pre-process
if not os.path.exists('celeba'):
    os.makedirs('celeba')
url = 'https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684'
output = 'celeba/data.zip'
gdown.download(url, output, quiet = False)

with ZipFile(output, 'r') as zipObj:
    zipObj.extractall('celeba')

dataset = keras.utils.image_dataset_from_directory(
    'celeba', label_mode = None, image_size = (64, 64), batch_size = 32
)
dataset = dataset.map(lambda x: x/255.0) # Normalization
for x in dataset:
    plt.axis('off')
    plt.imshow((x.numpy() * 255).astype('int32')[0])
    break

# Discriminator
D = keras.Sequential([
    keras.Input(shape = (64, 64, 3)),
    layers.Conv2D(64, kernel_size = 4, strides = 2, padding = 'same'),
    layers.LeakyReLU(negative_slope = 0.2),
    layers.Conv2D(128, kernel_size = 4, strides = 2, padding = 'same'),
    layers.LeakyReLU(negative_slope = 0.2),
    layers.Conv2D(128, kernel_size = 4, strides = 2, padding = 'same'),
    layers.LeakyReLU(negative_slope = 0.2),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(1, activation = 'sigmoid')
], name = 'Discriminator')

D.summary()

# Generator
latentZDim = 128
G = keras.Sequential([
    keras.Input(shape = (latentZDim,)),
    layers.Dense(8 * 8 * 128),
    layers.Reshape((8, 8, 128)),
    layers.Conv2DTranspose(128, kernel_size = 4, strides = 2, padding = 'same'),
    layers.LeakyReLU(negative_slope = 0.2),
    layers.Conv2DTranspose(256, kernel_size = 4, strides = 2, padding = 'same'),
    layers.LeakyReLU(negative_slope = 0.2),
    layers.Conv2DTranspose(512, kernel_size = 4, strides = 2, padding = 'same'),
    layers.LeakyReLU(negative_slope = 0.2),
    layers.Conv2D(3, kernel_size = 3, activation = 'sigmoid', padding = 'same'),
], name = 'Generator')
G.summary()

# Override Train Step
class GAN(keras.Model):
    def __init__(self, dis, gen, zDim) -> None:
        super().__init__()
        self.dis = dis
        self.gen = gen
        self.zDim = zDim
        self.seedGenerator = keras.random.SeedGenerator(1337)

    def compile(self, dOptim, gOptim, lossFunc):
        super().compile(loss = lossFunc)
        self.dOptim = dOptim
        self.gOptim = gOptim
        self.lossFunc = lossFunc
        self.dLossMetric = keras.metrics.Mean(name = 'DLoss')
        self.gLossMetric = keras.metrics.Mean(name = 'GLoss')

    @property
    def metrics(self):
        return [self.dLossMetric, self.gLossMetric]

    def call(self, inputs):
        # This method should define the forward pass of your GAN
        # For example, you might generate fake images and return them:
        batchSize = ops.shape(inputs)[0]
        latentNoise = keras.random.normal(
            shape = (batchSize, self.zDim), seed = self.seedGenerator
        )
        return self.gen(latentNoise)

    def train_step(self, data):
        realImgs = data
        batchSize = ops.shape(realImgs)[0]
        latentNoise = keras.random.normal(
            shape = (batchSize, self.zDim), seed = self.seedGenerator
        )
        # Here we're generating fake images in batches
        genImgs = self.gen(latentNoise)
        # Concat them with real images
        combinedImgs = ops.concatenate([genImgs, realImgs], axis = 0)
        labels = ops.concatenate([ops.ones((batchSize, 1)), ops.zeros((batchSize, 1))],
                                 axis = 0) # Labeling 1 for gen and 0 for real
        labels += 0.05 * tf.random.uniform(tf.shape(labels)) # !?adding noise trick?!

        # Train Discriminator
        with tf.GradientTape() as tape:
            preds = self.dis(combinedImgs)
            dLoss = self.lossFunc(labels, preds)

        grads = tape.gradient(dLoss, self.dis.trainable_weights)
        self.dOptim.apply_gradients(zip(grads, self.dis.trainable_weights))

        # Train Generator
        latentNoise = keras.random.normal(
            shape = (batchSize, self.zDim), seed = self.seedGenerator
        )
        misleadingLabels = ops.zeros((batchSize, 1)) # Says: All is REAL!

        with tf.GradientTape() as tape:
            preds = self.dis(self.gen(latentNoise))
            gLoss = self.lossFunc(misleadingLabels, preds)

        grads = tape.gradient(gLoss, self.gen.trainable_weights)
        self.gOptim.apply_gradients(zip(grads, self.gen.trainable_weights))

        # Update Metrics
        self.dLossMetric.update_state(dLoss)
        self.gLossMetric.update_state(gLoss)
        return {
            'DLoss': self.dLossMetric.result(),
            'GLoss': self.gLossMetric.result(),
        }

# Callback
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, numImgs = 3, latentDim = 128):
        self.numImgs = numImgs
        self.latentDim = latentDim
        self.seedGenerator = keras.random.SeedGenerator(42)

    def on_epoch_end(self, epoch, logs = None):
        randomLatentVector = keras.random.normal(
            shape = (self.numImgs, self.latentDim), seed = self.seedGenerator
        )
        genImgs = self.model.gen(randomLatentVector)
        genImgs *= 255
        genImgs.numpy()
        for i in range(self.numImgs):
            img = keras.utils.array_to_img(genImgs[i])
            img.save(f'gen_img_{epoch}_{i}.png')

# Model fit
numEpochs = 10
gan = GAN(dis = D, gen = G, zDim = latentZDim)
gan.compile(
    dOptim = keras.optimizers.Adam(learning_rate = 0.0001),
    gOptim = keras.optimizers.Adam(learning_rate = 0.0001),
    lossFunc = keras.losses.BinaryCrossentropy(),
)
gan.fit(
    dataset, epochs = numEpochs,
    callbacks = [GANMonitor(numImgs = 10, latentDim = latentZDim)]
)
