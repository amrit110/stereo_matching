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

    Attributes:
        _device (str): 'cpu' or specific gpu.
        _logger (logging.Logger): python logger to log training progress.
        _exp_dir (str): path to experiment directory.
        _num_input_channels (int): number of input channels.
        _patch_feature_module (tf.keras.Sequential): patch feature extraction network.
        _global_step (tf.Variable): training step counter.
        _half_range (int): half range of disparity prediction window.
        _post_process (bool): flag to toggle post-processing.

    """

    def __init__(self, settings, device, exp_dir, logger, global_step):
        """Constructor."""
        super(SiameseStereoMatching, self).__init__()
        self._device = device
        self._logger = logger
        self._exp_dir = exp_dir
        self._num_input_channels = settings.num_input_channels
        self._global_step = global_step
        self._disparity_range = settings.disparity_range
        self._half_range = settings.half_range
        self._post_process = settings.post_process

        self.patch_feature_module = self._create_patch_feature_module(self._num_input_channels)

    def _create_patch_feature_module(self, num_input_channels):
        c = num_input_channels
        patch_feature_module = tf.keras.Sequential()

        patch_feature_module.add(self._create_conv_bn_relu_module(c, 32, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(32, 32, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(32, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))

        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5,
                                                                  add_relu=False))

        return patch_feature_module

    def _create_conv_bn_relu_module(self, num_input_channels, num_output_channels,
                                    kernel_height, kernel_width, add_relu=True):
        conv_bn_relu = tf.keras.Sequential()
        conv = tf.keras.layers.Conv2D(num_input_channels,
                                      (kernel_height, kernel_width),
                                      padding='valid',
                                      kernel_initializer=tf.initializers.he_uniform())
        bn = tf.keras.layers.BatchNormalization()
        relu = tf.keras.layers.ReLU()

        conv_bn_relu.add(conv)
        conv_bn_relu.add(bn)
        if add_relu:
            conv_bn_relu.add(relu)

        return conv_bn_relu

    def loss_fn(self, batch, training=None):
        """Loss function."""
        inner_product = self.call(batch.left_patches, batch.right_patches,
                                  training)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch.labels,
                                                                        logits=inner_product))

        return loss

    def grads_fn(self, batch, training=None):
        """Compute loss and gradients."""
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(batch, training)

        return tape.gradient(loss, self.variables), loss

    def plot_loss(self):
        """Plot current loss, and save figure to experiments folder."""
        fig = plt.figure()
        iterations_range = np.arange(100, self._global_step.numpy() + 1, 100)
        plt.plot(iterations_range, self.history['train_loss'],
                 color='b', label='Training loss')
        plt.plot(iterations_range, self.history['val_loss'],
                 color='r', label='Validation loss')
        plt.grid(True)
        plt.title('Loss plot during training', fontsize=15)
        plt.xlabel('Training Iteration', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig(join(self._exp_dir, 'loss.png'))
        plt.close(fig)

    def run_inference_on_test(self, testing_dataset):
        """Run model inference on test data, save outputs (qualitative results)."""
        itx = 0
        while True:
            try:
                self._logger.info('Running inference on test sample: {}'.format(itx))
                left_img, right_img = testing_dataset.iterator.get_next()
                disparity_prediction = self.inference(left_img, right_img)
                self.save_sample(disparity_prediction, left_img,
                                 right_img, itx)
                itx += 1
            except tf.errors.OutOfRangeError:
                self._logger.info('Reached end of testing dataset!')
                break

    def inference(self, left_image, right_image):
        """Run model on test images.

        Args:
            left_image (tf.Tensor): left input image.
            right_image (tf.Tensor): right input image.

        Returns:
            disp_prediction (tf.Tensor): disparity prediction.

        """
        cost_volume, win_indices = self.call(left_image, right_image, training=False,
                                             inference=True)
        img_height, img_width = cost_volume.shape[1], cost_volume.shape[2] # 1 x H x W x C
        if self._post_process:
            cost_volume = apply_cost_aggregation(cost_volume)
        cost_volume = tf.squeeze(cost_volume)
        row_indices, _ = tf.dtypes.cast(tf.meshgrid(tf.range(0, img_width),
                                                    tf.range(0, img_height)),
                                        dtype=tf.int64)

        disp_prediction_indices = tf.argmax(cost_volume, axis=-1)

        disp_prediction = []
        for i in range(img_width):
            column_disp_prediction = tf.gather(win_indices[i],
                                               disp_prediction_indices[:, i],
                                               axis=0)
            column_disp_prediction = tf.expand_dims(column_disp_prediction, 1)
            disp_prediction.append(column_disp_prediction)

        disp_prediction = tf.concat(disp_prediction, 1)
        disp_prediction = row_indices - disp_prediction

        return disp_prediction

    def save_sample(self, disp_prediction, left_image, right_image, iteration):
        """Save a test sample (inputs and prediction), during training loop.

        Args:
            disp_prediction (tf.Tensor): disparity prediction.
            left_image (tf.Tensor): left input image.
            right_image (tf.Tensor): right input image.
            iteration (int): training iteration count.

        """
        disp_img = np.array(disp_prediction)
        disp_img[disp_img < 0] = 0
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=disp_img.min(), vmax=disp_img.max())
        disp_img = cmap(norm(disp_img))
        left_img_save = np.array(left_image)[0]
        left_img_save = self._normalize_uint8(left_img_save)
        right_img_save = np.array(right_image)[0]
        right_img_save = self._normalize_uint8(right_img_save)
        self.save_images([left_img_save, disp_img], 1,
                         ['left image', 'disparity'], iteration)

    def _normalize_uint8(self, array):
        """Normalize image array and convert to uint8."""
        array = (array - array.min()) / (array.max() - array.min())
        array = (array * 255).astype(np.uint8)

        return array

    def save_images(self, images, cols, titles, iteration):
       """Save multiple images arranged as a table.

       Args:
           images (list): list of images to display as numpy arrays.
           cols (int): number of columns.
           titles (list): list of title strings for each image.
           iteration (int): iteration counter or plot interval.

       """
       assert((titles is None) or (len(images) == len(titles)))
       n_images = len(images)
       if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
       fig = plt.figure(figsize=(20, 10))
       for n, (image, title) in enumerate(zip(images, titles)):
           a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
           image = np.squeeze(image)
           if image.ndim == 2:
               plt.gray()
           plt.imshow(image)

           a.axis('off')
           a.set_title(title, fontsize=40)
       fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
       plt.savefig(join(self._exp_dir, 'qualitative_samples',
                        'output_sample_{}.png'.format(iteration)),
                   bbox_inches='tight')
       plt.close(fig)

    def fit(self, training_dataset, validation_dataset, optimizer,
            num_iterations):
        """Train and validate on dataset.

        Args:
            training_dataset (Dataset): training dataset object.
            validation_dataset (Dataset): validation dataset object.
            optimizer (tf.train.Optimizer): optimizer.
            num_iterations (int): number of training iterations.

        """
        # Initialize objects update the mean loss of train and eval.
        train_loss = tfe.metrics.Mean('train_loss')
        val_loss = tfe.metrics.Mean('val_loss')

        # Initialize dictionary to store the loss history.
        self.history = {}
        self.history['train_loss'] = []
        self.history['val_loss'] = []

        itx = 0
        # NOTE: Hard-coded to run 100 iterations of training, then validation.
        # Ideally we can also have 1 iteration of training and validation but
        # this setup allows initial batch-norm statistics to be more accurate, so
        # that the validation loss in the first few iterations is also
        # meaningful.
        fixed_iters = 100
        with tf.device(self._device):
            while itx < num_iterations:
                # NOTE: Ok, so this seems ugly, but apparently tfe does not
                # support re-initializable dataset iterators yet. So, need to fix
                # this later.
                try:
                    # Training iterations.
                    for i in range(fixed_iters):
                        left_patches, right_patches, labels = training_dataset.iterator.get_next()
                        batch = Batch(left_patches, right_patches, labels)
                        grads, t_loss = self.grads_fn(batch, training=True)
                        optimizer.apply_gradients(zip(grads, self.variables))
                        train_loss(t_loss)

                    # Validation iterations.
                    for i in range(fixed_iters):
                        left_patches, right_patches, labels = validation_dataset.iterator.get_next()
                        batch = Batch(left_patches, right_patches, labels)
                        v_loss = self.loss_fn(batch, training=False)
                        val_loss(v_loss)

                    itx += fixed_iters
                    self._global_step.assign_add(fixed_iters)

                    self.history['train_loss'].append(train_loss.result().numpy())
                    self.history['val_loss'].append(val_loss.result().numpy())
                    self.plot_loss()

                    random_img_idx = np.random.choice(validation_dataset.sample_ids)
                    disparity_prediction = self._run_inference_single(validation_dataset,
                                                                      random_img_idx)
                    self.save_sample(disparity_prediction, sample_left_img,
                                     sample_right_img, itx)

                    # Print train and eval losses.
                    self._logger.info('Train loss at iteration {}: {:04f}'.\
                                      format(itx+1, train_loss.result().numpy()))
                    self._logger.info('Validation loss at iteration {}: {:04f}'.\
                                     format(itx+1, val_loss.result().numpy()))

                    # Save checkpoint.
                    self.save_model()
                except tf.errors.OutOfRangeError:
                    break

    def _run_inference_single(self, validation_dataset, idx):
        """Run inference on a validation set image."""
        paddings = validation_dataset.get_paddings()
        left_img = tf.pad(tf.expand_dims(validation_dataset.left_images[idx], 0),
                          paddings, "CONSTANT")
        right_img = tf.pad(tf.expand_dims(validation_dataset.right_images[idx], 0),
                           paddings, "CONSTANT")
        disparity_prediction = self.inference(left_img, right_img)

        return disparity_prediction

    def run_inference_val(self, validation_dataset):
        """Run inference on validation set, compute 3-pixel error metric.

        Args:
            validation_dataset (Dataset): validation dataset object.

        Returns:
            mean_error (float): 3-pixel mean error over validation set.

        """
        error = 0
        self._logger.info('Computing error on validation set ...')
        for i, idx in enumerate(validation_dataset.sample_ids):
            disparity_prediction = self._run_inference_single(validation_dataset, idx)
            disparity_ground_truth  = validation_dataset.disparity_images[idx]

            valid_gt_pixels = (disparity_ground_truth != 0).astype('float')
            masked_prediction_valid = disparity_prediction * valid_gt_pixels
            num_valid_gt_pixels = valid_gt_pixels.sum()

            # NOTE: Use 3-pixel error metric for now.
            num_error_pixels = (np.abs(masked_prediction_valid -
                                  disparity_ground_truth) > 3).sum()
            error += num_error_pixels / num_valid_gt_pixels

            self._logger.info('Error sum {:04f}, {} samples evaluated ...'.format(error, i+1))

        mean_error = error / len(validation_dataset.sample_ids)

        return mean_error

    def call(self, left_input, right_input, training=None, mask=None, inference=False):
        """Forward pass call.

        During training, we compute inner product between left patch feature
        and all features from right patch. During testing, we compute a batch
        matrix multiplication to compute inner-product over entire image and
        obtain a cost volume.

        Args:
            left_input (tf.Tensor): left image input.
            right_input (tf.Tensor): right image input.
            training (bool, optional): flag to toggle training vs. test mode
            which affects batch-norm layers.
            mask (, optional): mask
            inference (bool, optional): used for inference mode, affects the
            way inner product is computed.

        Returns:
            inner_product (tf.Tensor): output from feature matching model, size
            of tensor varies between training and inference.

        """
        left_feature = self.patch_feature_module(left_input, training=training)
        right_feature = self.patch_feature_module(right_input, training=training)

        if inference:
            inner_product, win_indices = [], []
            img_height, img_width = right_feature.shape[1], right_feature.shape[2]
            row_indices = tf.dtypes.cast(tf.range(0, img_width), dtype=tf.int64)

            for i in range(img_width):
                left_column_features = tf.squeeze(left_feature[:, :, i])
                start_win = max(0, i - self._half_range)
                end_win = max(self._disparity_range, self._half_range + i + 1)
                start_win = start_win - max(0, end_win - img_width.value)
                end_win = min(img_width, end_win)

                right_win_features = tf.squeeze(right_feature[:, :, start_win:end_win])
                win_indices_column = tf.expand_dims(row_indices[start_win:end_win], 0)
                inner_product_column = tf.einsum('ij,ikj->ik', left_column_features,
                                                 right_win_features)
                inner_product_column = tf.expand_dims(inner_product_column, 1)
                inner_product.append(inner_product_column)
                win_indices.append(win_indices_column)

            inner_product = tf.expand_dims(tf.concat(inner_product, 1), 0)
            win_indices = tf.concat(win_indices, 0)

            return inner_product, win_indices
        else:
            left_feature = tf.squeeze(left_feature)
            inner_product = tf.einsum('il,ijkl->ijk', left_feature, right_feature)

            return inner_product

    def restore_model(self, checkpoint_name=None):
        """ Function to restore trained model."""
        with tf.device(self._device):
            dummy_input = tf.constant(tf.zeros((5, 37, 37, self._num_input_channels)))
            dummy_pred = self.call(dummy_input, dummy_input, training=False)
            checkpoint_path = join(self._exp_dir, 'checkpoints')
            if checkpoint_name is None:
                checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            else:
                checkpoint_path = join(checkpoint_path, checkpoint_name)
            tfe.Saver(self.variables).restore(checkpoint_path)

    def save_model(self):
        """ Function to save trained model."""
        tfe.Saver(self.variables).save(join(self._exp_dir, 'checkpoints', 'checkpoints'),
                                       global_step=self._global_step)


def apply_cost_aggregation(cost_volume):
    """Apply cost-aggregation post-processing to network predictions.

    Performs an average pooling operation over raw network predictions to smoothen
    the output.

    Args:
        cost_volume (tf.Tensor): cost volume predictions from network.

    Returns:
        cost_volume (tf.Tensor): aggregated cost volume.

    """
    # NOTE: Not ideal but better than zero padding, since we average.
    cost_volume = tf.pad(cost_volume, tf.constant([[0, 0,], [2, 2,], [2, 2], [0, 0]]),
                         "REFLECT")

    return tf.layers.average_pooling2d(cost_volume,
                                       pool_size=(5, 5),
                                       strides=(1, 1),
                                       padding='VALID',
                                       data_format='channels_last')
