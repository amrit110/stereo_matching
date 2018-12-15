"""Data pre-processing and loading."""

import os
import glob
import pickle

import numpy as np
import tensorflow as tf

from pre_process import *


def load_image(image_path):
    image_file = tf.read_file(image_path)
    image = tf.image.decode_png(image_file)
    image = image[0:IMG_HEIGHT, 0:IMG_WIDTH]

    return image


def load_images(left_image_paths, right_image_paths):
    left_images = []
    right_images = []
    for idx in range(left_image_paths.shape[0]):
        left_images.append(load_image(left_image_paths[idx]))
        right_images.append(load_image(left_image_paths[idx]))

    return tf.convert_to_tensor(left_images), tf.convert_to_tensor(right_images)


def _parse_function(sample_info):
    # Unpack.
    idx = tf.to_int32(sample_info[0])
    left_center_x = tf.to_int32(sample_info[1])
    left_center_y = tf.to_int32(sample_info[2])
    right_center_x = tf.to_int32(sample_info[3])

    # left_image_path = left_image_paths[idx]
    # left_image = load_image(left_image_path)
    left_image = left_images[idx]

    # right_image_path = right_image_paths[idx]
    # right_image = load_image(right_image_path)
    right_image = right_images[idx]

    left_patch = left_image[left_center_y - HALF_PATCH_SIZE:left_center_y + HALF_PATCH_SIZE + 1,
                            left_center_x - HALF_PATCH_SIZE:left_center_x + HALF_PATCH_SIZE + 1, :]
    right_patch = right_image[left_center_y - HALF_PATCH_SIZE:left_center_y + HALF_PATCH_SIZE + 1,
                              right_center_x - HALF_PATCH_SIZE - HALF_RANGE:right_center_x + \
                              HALF_PATCH_SIZE + HALF_RANGE + 1, :]

    return left_patch, right_patch


def create_dataset(valid_locations):
    valid_locations_train = valid_locations['valid_locations_train']

    dataset = tf.data.Dataset.from_tensor_slices(valid_locations_train)

    dataset = dataset.map(_parse_function)
    batched_dataset = dataset.batch(128)

    iterator = batched_dataset.make_one_shot_iterator()

    for i in range(100):
        next_element = iterator.get_next()
        print(next_element[0].shape, next_element[1].shape)

if __name__=='__main__':
    tf.enable_eager_execution()
    with open('cache/kitti_2015/training/patch_locations.pkl', 'rb') as handle:
        valid_locations = pickle.load(handle)

    left_image_paths, right_image_paths, _ = \
            [tf.constant(paths) for paths in load_image_paths()]
    left_images, right_images = load_images(left_image_paths,
                                            right_image_paths)
    dataset = create_dataset(valid_locations)
