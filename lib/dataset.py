"""Data pre-processing and loading."""

# Imports.
import os
import glob
import pickle
from PIL import Image

import numpy as np
import tensorflow as tf

from lib.pre_process import load_image_paths
from lib.utils import trim_image


def _load_image(image_path, img_height, img_width):
    """Load image as tf.Tensor.

    Args:
        image_path (str): path to image.
        img_height (int): desired height of output image (excess trimmed).
        img_width (int): desired width of output image (excess trimmed).

    Returns:
        img (tf.Tensor): image array as tensor.

    """
    image_file = tf.read_file(image_path)
    img = tf.image.decode_png(image_file)
    img = trim_image(img, img_height, img_width)
    img = tf.image.per_image_standardization(img)

    return img


def _load_disparity(image_path, img_height, img_width):
    """Load disparity image as numpy array.

    Args:
        image_path (str): path to disparity image.
        img_height (int): desired height of output image (excess trimmed).
        img_width (int): desired width of output image (excess trimmed).

    Returns:
        disp_img (numpy.ndarray): disparity image array as tensor.

    """
    disp_img = np.array(Image.open(image_path)).astype('float64')
    disp_img = trim_image(disp_img, img_height, img_width)
    disp_img /= 256

    return disp_img


def _load_images(left_image_paths, right_image_paths, disparity_paths, img_height, img_width):
    left_images = []
    right_images = []
    disparity_images = []
    for idx in range(left_image_paths.shape[0]):
        left_images.append(_load_image(left_image_paths[idx], img_height, img_width))
        right_images.append(_load_image(right_image_paths[idx], img_height, img_width))

        if disparity_paths:
            disparity_images.append(_load_disparity(disparity_paths[idx], img_height, img_width))

    return (tf.convert_to_tensor(left_images),
            tf.convert_to_tensor(right_images),
            np.array(disparity_images))


def _get_labels(disparity_range, half_range):
    gt = np.zeros((disparity_range))

    # NOTE: Smooth targets are [0.05, 0.2, 0.5, 0.2, 0.05], hard-coded.
    gt[half_range - 2: half_range + 3] = np.array([0.05, 0.2, 0.5, 0.2, 0.05])

    return gt


class Dataset:
    """Dataset class to provide training and validation data.

    When initialized, loads patch locations info from file, loads all left and
    right camera images into memory for enabling fast loading.

    Attributes:
        left_images (tf.Tensor): tensor of all left camera images.
        right_images (tf.Tensor): tensor of all right camera images.
        iterator (tf.data.Iterator): iterator over patches dataset.

    """

    def __init__(self, settings, patch_locations, phase):
        """Constructor.

        Args:
            settings (argparse.Namespace): settings for the project derived from
            main script.
            patch_locations (dict): dict with arrays containing patch locations
            info.
            phase (str): 'train' or 'val' phase.

        """
        self._settings = settings
        left_image_paths, right_image_paths, disparity_paths = \
                load_image_paths(settings.data_path,
                                 settings.left_img_folder,
                                 settings.right_img_folder,
                                 settings.disparity_folder)
        left_image_paths = tf.constant(left_image_paths)
        right_image_paths = tf.constant(right_image_paths)
        self.left_images, self.right_images, self.disparity_images = \
                _load_images(left_image_paths,
                             right_image_paths,
                             disparity_paths,
                             settings.img_height,
                             settings.img_width)
        self.iterator = self._create_dataset_iterator(patch_locations, phase)
        if phase == 'train':
            self.sample_ids = patch_locations['train_ids']
        elif phase == 'val':
            self.sample_ids = patch_locations['val_ids']

    def get_paddings(self):
        """Zero padding used for inference.

        Returns:
            (tf.Tensor): tensor specifying zero-padding to apply to image
            during inference.

        """
        return tf.constant([[0, 0,],
                            [self._settings.half_patch_size, self._settings.half_patch_size],
                            [self._settings.half_patch_size, self._settings.half_patch_size],
                            [0, 0]])

    def _parse_function(self, sample_info):
        """Parsing function passed to map operation for loading data."""
        idx = tf.to_int32(sample_info[0])
        left_center_x = tf.to_int32(sample_info[1])
        left_center_y = tf.to_int32(sample_info[2])
        right_center_x = tf.to_int32(sample_info[3])

        left_image = self.left_images[idx]
        right_image = self.right_images[idx]

        left_patch = left_image[left_center_y -\
                                self._settings.half_patch_size:left_center_y +\
                                self._settings.half_patch_size + 1,
                                left_center_x -\
                                self._settings.half_patch_size:left_center_x +\
                                self._settings.half_patch_size + 1, :]
        right_patch = right_image[left_center_y -\
                                  self._settings.half_patch_size:left_center_y +\
                                  self._settings.half_patch_size + 1,
                                  right_center_x - self._settings.half_patch_size -\
                                  self._settings.half_range:right_center_x +\
                                  self._settings.half_patch_size +\
                                  self._settings.half_range + 1, :]


        labels = tf.convert_to_tensor(_get_labels(self._settings.disparity_range,
                                                  self._settings.half_range))

        return left_patch, right_patch, labels

    def _test_parse_function(self, idx):
        """Parsing function passed to map operation for loading data (testing)."""
        left_image = self.left_images[idx]
        right_image = self.right_images[idx]

        paddings = self.get_paddings()
        left_image = tf.pad(tf.expand_dims(left_image, 0), paddings, "CONSTANT")
        right_image = tf.pad(tf.expand_dims(right_image, 0), paddings, "CONSTANT")

        return left_image, right_image


    def _create_train_iterator(self, patch_locations):
        """Create training data iterator."""
        dataset_locations = patch_locations['valid_locations_train']

        dataset = tf.data.Dataset.from_tensor_slices(dataset_locations)
        dataset = dataset.map(self._parse_function)
        batched_dataset = dataset.batch(self._settings.batch_size)
        iterator = batched_dataset.make_one_shot_iterator()

        return iterator

    def _create_val_iterator(self, patch_locations):
        """Create validation data iterator."""
        dataset_locations = patch_locations['valid_locations_val']
        # NOTE: Repeat dataset so that we can have 40k iterations.
        dataset_locations = dataset_locations.repeat(2, axis=0)

        dataset = tf.data.Dataset.from_tensor_slices(dataset_locations)
        dataset = dataset.map(self._parse_function)
        batched_dataset = dataset.batch(self._settings.batch_size)
        iterator = batched_dataset.make_one_shot_iterator()

        return iterator

    def _create_test_iterator(self):
        """Create testing dataset iterator for inference."""
        input_ids = tf.range(self.left_images.shape[0])
        dataset = tf.data.Dataset.from_tensor_slices(input_ids)
        # NOTE: Loads 1 sample, i.e. batch mode not implemented yet.
        dataset = dataset.map(self._test_parse_function)
        iterator = dataset.make_one_shot_iterator()

        return iterator

    def _create_dataset_iterator(self, patch_locations, phase='train'):
        """Create dataset iterator."""
        if phase == 'train':
            return self._create_train_iterator(patch_locations)
        elif phase == 'val':
            return self._create_val_iterator(patch_locations)
        elif phase == 'test':
             return self._create_test_iterator()
