"""Model defintion."""

# Imports.
import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe


tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)


class CNN(tf.keras.Model):

    def __init__(self, num_input_channels):
        super(CNN, self).__init__()
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

    def call(self, inputs, training=None, mask=None):
        left_feature = self.patch_feature_module(inputs['left_patch'])
        right_feature = self.patch_feature_module(inputs['right_patch'])

        print(left_feature.shape, right_feature.shape)

        inner_product = self.inner_product_layer([left_feature, right_feature])
        inner_product = tf.squeeze(inner_product, squeeze_dims=[1, 2])

        return inner_product



if __name__ == '__main__':
    device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

    with tf.device(device):
        model = CNN(1)

        dummy_left_patch = tf.zeros((2, 37, 37, 1))
        dummy_right_patch = tf.zeros((2, 37, 237, 1))

        inputs = {'left_patch': dummy_left_patch,
                  'right_patch': dummy_right_patch}
        outputs = model(inputs)
        print(outputs.shape)
