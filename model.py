"""Model defintion."""

# Imports.
import os
import numpy as np
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe


Batch = namedtuple('Batch', ['left_patches', 'right_patches', 'labels'])


class SiameseStereoMatching(tf.keras.Model):

    def __init__(self, num_input_channels, device):
        super(SiameseStereoMatching, self).__init__()
        self.device = device

        self.patch_feature_module = self.create_patch_feature_module(num_input_channels)
        self.inner_product_layer = tf.keras.layers.Dot(axes=-1)

    def create_patch_feature_module(self, num_input_channels):
        c = num_input_channels
        patch_feature_module = tf.keras.Sequential()

        patch_feature_module.add(self.create_conv_bn_relu_module(c, 32, 5, 5))
        patch_feature_module.add(self.create_conv_bn_relu_module(32, 32, 5, 5))
        patch_feature_module.add(self.create_conv_bn_relu_module(32, 64, 5, 5))
        patch_feature_module.add(self.create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self.create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self.create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self.create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self.create_conv_bn_relu_module(64, 64, 5, 5))

        patch_feature_module.add(self.create_conv_bn_relu_module(64, 64, 5, 5,
                                                                 add_relu=False))

        return patch_feature_module

    def create_conv_bn_relu_module(self, num_input_channels, num_output_channels,
                                   kernel_height, kernel_width, add_relu=True):
        conv_bn_relu = tf.keras.Sequential()
        conv = tf.keras.layers.Conv2D(num_input_channels,
                                      (kernel_height, kernel_width),
                                      padding='valid',
                                      kernel_initializer='truncated_normal')
        bn = tf.keras.layers.BatchNormalization()
        relu = tf.keras.layers.ReLU()

        conv_bn_relu.add(conv)
        conv_bn_relu.add(bn)
        if add_relu:
            conv_bn_relu.add(relu)

        return conv_bn_relu

    def loss_fn(self, batch, training=None):
        inner_product = self.call(batch.left_patches, batch.right_patches,
                                  training)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch.labels,
                                                                        logits=inner_product))

        return loss

    def fit(self, training_dataset, validation_dataset, optimizer,
            num_iterations):
        # Initialize classes to update the mean loss of train and eval.
        train_loss = tfe.metrics.Mean('train_loss')
        val_loss = tfe.metrics.Mean('val_loss')

        # Initialize dictionary to store the loss history.
        self.history = {}
        self.history['train_loss'] = []
        self.history['val_loss'] = []

        with tf.device(self.device):
            for i in range(num_iterations):
                # Training iteration.
                left_patches, right_patches, labels = training_dataset.iterator.get_next()
                batch = Batch(left_patches, right_patches, labels)
                grads, loss = self.grads_fn(batch, training=True)
                optimizer.apply_gradients(zip(grads, self.variables))
                train_loss(loss)
                self.history['train_loss'].append(train_loss.result().numpy())

                # Validation iteration.
                left_patches, right_patches, labels = validation_dataset.iterator.get_next()
                batch = Batch(left_patches, right_patches, labels)
                loss = self.loss_fn(batch, training=False)
                val_loss(loss)
                self.history['val_loss'].append(val_loss.result().numpy())

                # Print train and eval losses.
                if (i==0) | ((i+1) % args.verbose == 0):
                    print('Train loss at epoch %d: ' %(i+1), self.history['train_loss'][-1])
                    print('Eval loss at epoch %d: ' %(i+1), self.history['eval_loss'][-1])

    def grads_fn(self, batch, training=None):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(batch, training)

        return tape.gradient(loss, self.variables), loss

    def call(self, left_input, right_input, training=None, mask=None):
        left_feature = self.patch_feature_module(left_input)
        right_feature = self.patch_feature_module(right_input)

        inner_product = self.inner_product_layer([left_feature, right_feature])
        inner_product = tf.squeeze(inner_product, squeeze_dims=[1, 2])

        return inner_product



if __name__ == '__main__':
    tf.enable_eager_execution()

    tf.set_random_seed(0)
    np.random.seed(0)
    device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

    with tf.device(device):
        model = SiameseStereoMatching(1)

        dummy_left_patch = tf.zeros((2, 37, 37, 1))
        dummy_right_patch = tf.zeros((2, 37, 237, 1))

        outputs = model(dummy_left_patch, dummy_right_patch)
        print(outputs.shape)
