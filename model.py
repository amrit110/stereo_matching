"""Stereo-Matching Model."""

# Imports.
import os
from os.path import join
import numpy as np
from collections import namedtuple
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
import keras.backend as K


Batch = namedtuple('Batch', ['left_patches', 'right_patches', 'labels'])


class SiameseStereoMatching(tf.keras.Model):
    """Model implementation.
       Implementation of Luo, W., & Schwing, A. G. (n.d.). Efficient Deep Learning for Stereo Matching.
    
    """

    def __init__(self, num_input_channels, device, exp_dir, logger):
        super(SiameseStereoMatching, self).__init__()
        self.device = device
        self.logger = logger
        self.exp_dir = exp_dir

        self.patch_feature_module = self.create_patch_feature_module(num_input_channels)

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

    def grads_fn(self, batch, training=None):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(batch, training)

        return tape.gradient(loss, self.variables), loss

    def plot_loss(self):
        fig = plt.figure()
        plt.plot(range(len(self.history['train_loss'])), self.history['train_loss'],
                 color='b', label='Training loss')
        plt.plot(range(len(self.history['val_loss'])), self.history['val_loss'], 
         	 color='r', label='Validation loss')
        plt.grid(True)
        plt.title('Loss plot during training', fontsize=15)
        plt.xlabel('Plot interval step', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig(join(self.exp_dir, 'loss.png'))
        plt.close(fig)

    def inference(self, left_image, right_image):
        outputs = self.call(left_image, right_image, training=False, inference=True)
        outputs = tf.squeeze(tf.argmax(outputs, axis=-1))
        row_indices, _ = tf.meshgrid(tf.range(0, outputs.shape[1]),
                                     tf.range(0, outputs.shape[0]))
        disp_prediction = row_indices - tf.dtypes.cast(outputs, dtype=tf.int32)
        
        return disp_prediction

    def save_sample(self, disp_prediction, left_image, right_image, iteration):
        disp_img = np.array(disp_prediction)
        disp_img = (disp_img * (disp_img >= 0))
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=disp_img.min(), vmax=disp_img.max()) 
        disp_img = cmap(norm(disp_img))
        left_img_save = np.array(left_image)[0]
        left_img_save = self.normalize_uint8(left_img_save)
        right_img_save = np.array(right_image)[0]
        right_img_save = self.normalize_uint8(right_img_save)
        self.save_images([left_img_save, 
                          right_img_save,
                          disp_img], 2, 
                         ['left image', 'right image', 'disparity'],
                         iteration)

    def normalize_uint8(self, array):
        array = (array - array.min()) / (array.max() - array.min())
        array = (array * 255).astype(np.uint8)

        return array

    def save_images(self, images, cols, titles, iteration):
       """Save multiple images arranged as a table.

       Args:
           images (list): list of images to display as numpy arrays.
           cols (int): number of columns.
           titles (list): list of title strings for each image.

       """
       assert((titles is None) or (len(images) == len(titles)))
       n_images = len(images)
       if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
       fig = plt.figure(figsize=(20, 10))
       for n, (image, title) in enumerate(zip(images, titles)):
           a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
           if image.ndim == 2:
               plt.gray()
           plt.imshow(image)
           a.set_title(title)
       fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
       plt.savefig(join(self.exp_dir, 'output_sample_{}.png'.format(iteration)), 
                   bbox_inches='tight', pad_inches=0)
       plt.close(fig)

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
                grads, t_loss = self.grads_fn(batch, training=True)
                optimizer.apply_gradients(zip(grads, self.variables))
                train_loss(t_loss)

                # Validation iteration.
                left_patches, right_patches, labels = validation_dataset.iterator.get_next()
                batch = Batch(left_patches, right_patches, labels)
                v_loss = self.loss_fn(batch, training=False)
                val_loss(v_loss)

                if (i+1) % 100 == 0:
                    self.history['train_loss'].append(train_loss.result().numpy())
                    self.history['val_loss'].append(val_loss.result().numpy())
                    self.plot_loss()

                    paddings = tf.constant([[0, 0,], [18, 18], [18, 18], [0, 0]])
                    random_img_idx = np.random.randint(0, validation_dataset.left_images.shape[0]) 
                    sample_left_img = tf.pad(tf.expand_dims(validation_dataset.left_images[random_img_idx], 0), 
                                             paddings, "CONSTANT")
                    sample_right_img = tf.pad(tf.expand_dims(validation_dataset.right_images[random_img_idx], 0),
                                              paddings, "CONSTANT")
                    disparity_prediction = self.inference(sample_left_img, sample_right_img)
                    self.save_sample(disparity_prediction, sample_left_img, sample_right_img, i) 

                # Print train and eval losses.
                self.logger.info('Train loss at iteration {}: {:04f}'.format(i+1, train_loss.result().numpy()))
                self.logger.info('Validation loss at iteration {}: {:04f}'.format(i+1, val_loss.result().numpy()))

    def call(self, left_input, right_input, training=None, mask=None, inference=False):
        left_feature = self.patch_feature_module(left_input, training=training)
        right_feature = self.patch_feature_module(right_input, training=training)
        if inference:
            inner_product = tf.einsum('ijkl,ijnl->ijkn', left_feature, right_feature)
        else:

            left_feature = tf.squeeze(left_feature)
            inner_product = tf.einsum('il,ijkl->ijk', left_feature, right_feature)

        return inner_product



if __name__ == '__main__':
    tf.enable_eager_execution()

    tf.set_random_seed(0)
    np.random.seed(0)
    device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

    with tf.device(device):
        model = SiameseStereoMatching(3, device, None, None)
        # a, b = tf.meshgrid(tf.range(0, 1224), tf.range(0, 370))
        # print(a)
        # print(b)
        dummy_left_patch = tf.zeros((5, 37, 37, 3))
        # dummy_left_patch = tf.zeros((5, 406, 1260, 3))
        dummy_right_patch = tf.zeros((5, 406, 1260, 3))
        # dummy_left_patch = tf.transpose(dummy_left_patch, [0, 3, 1, 2])
        # dummy_right_patch = tf.transpose(dummy_right_patch, [0, 3, 1, 2])
        # paddings = tf.constant([[0, 0,], [18, 18], [18, 18], [0, 0]])
        # dummy_left_patch = tf.pad(dummy_left_patch, paddings, "CONSTANT")
        # dummy_right_patch = tf.pad(dummy_right_patch, paddings, "CONSTANT")
        print(dummy_right_patch.shape)
        outputs = model(dummy_left_patch, dummy_right_patch, training=False, inference=False)
        print(outputs.shape)
