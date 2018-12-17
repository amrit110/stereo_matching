"""Visualize KITTI stereo and disparity images. Pre-process to create dataset
to sample mini-batches for training."""


# Imports.
import os
import random
from random import shuffle
import glob
import pickle
from os.path import join

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import *


def load_image_paths(data_path, left_img_folder, right_img_folder,
                     disparity_folder):
    left_image_paths = sorted(glob.glob(join(data_path, left_img_folder, '*10.png')))
    right_image_paths = sorted(glob.glob(join(data_path, right_img_folder, '*10.png')))
    disparity_image_paths = sorted(glob.glob(join(data_path, disparity_folder, '*10.png')))

    return left_image_paths, right_image_paths, disparity_image_paths


def _is_valid_location(sample_locations, img_width, img_height,
                       half_patch_size, half_range):
    left_center_x, left_center_y, right_center_x, right_center_y = sample_locations
    is_valid_loc_left = ((left_center_x + half_patch_size + 1) <= img_width) and \
            ((left_center_x - half_patch_size) >= 0) and \
            ((left_center_y + half_patch_size + 1) <= img_height) and \
            ((left_center_y - half_patch_size) >= 0)
    is_valid_loc_right = ((right_center_x - half_range - half_patch_size) >= 0) and \
            ((right_center_x + half_range + half_patch_size + 1) <= img_width) and \
            ((right_center_y - half_patch_size) >= 0) and \
            ((right_center_y + half_patch_size + 1) <= img_height)

    return (is_valid_loc_left and is_valid_loc_right)


def compute_valid_locations(disparity_image_paths, sample_ids, img_height,
                            img_width, half_patch_size, half_range):
    num_samples = len(sample_ids)
    num_valid_locations = np.zeros(num_samples)

    for i, idx in enumerate(sample_ids):
        disp_img = np.array(Image.open(disparity_image_paths[idx])).astype('float64')
        # We want images of same size for efficient loading.
        disp_img = trim_image(disp_img, img_height, img_width)
        disp_img /= 256
        num_valid_locations[i] = (disp_img != 0).sum()

    num_valid_locations = int(num_valid_locations.sum())
    valid_locations = np.zeros((num_valid_locations, 4))
    valid_count = 0

    for i, idx in enumerate(sample_ids):
        disp_img = np.array(Image.open(disparity_image_paths[idx])).astype('float64')
        # We want images of same size for efficient loading.
        disp_img = disp_img[0:img_height, 0:img_width]
        disp_img /= 256
        row_locs, col_locs = np.where(disp_img != 0)
        # NOTE: Remove later.
        img_height, img_width = disp_img.shape

        for j, row_loc in enumerate(row_locs):
            left_center_x = col_locs[j]
            left_center_y = row_loc
            right_center_x = int(round(col_locs[j] - disp_img[left_center_y,
                                                              left_center_x]))
            right_center_y = left_center_y # Stereo pair is assumed to be rectified.

            sample_locations = (left_center_x, left_center_y, right_center_x, right_center_y)
            if _is_valid_location(sample_locations, img_width, img_height,
                                  half_patch_size, half_range):
                location_info = np.array([idx, left_center_x,
                                          left_center_y,
                                          right_center_x])
                valid_locations[valid_count, :] = location_info
                valid_count += 1

    valid_locations = valid_locations[0:valid_count, :]
    np.random.shuffle(valid_locations) # Shuffle samples.

    return valid_locations


def find_and_store_patch_locations(settings):
    left_image_paths, right_image_paths, disparity_image_paths = \
            load_image_paths(settings.data_path, settings.left_img_folder,
                             settings.left_img_folder, settings.disparity_folder)
    sample_indices = list(range(len(left_image_paths)))
    shuffle(sample_indices)
    train_ids = sample_indices[0:settings.num_train]
    val_ids = sample_indices[settings.num_train:]

    # Training set.
    valid_locations_train = compute_valid_locations(disparity_image_paths,
                                                    train_ids,
                                                    settings.img_height,
                                                    settings.img_width,
                                                    settings.half_patch_size,
                                                    settings.half_range)
    # Validation set.
    valid_locations_val = compute_valid_locations(disparity_image_paths,
                                                  val_ids,
                                                  settings.img_height,
                                                  settings.img_width,
                                                  settings.half_patch_size,
                                                  settings.half_range)

    # Save to disk
    contents_to_save = {'sample_indices': sample_indices,
                        'train_ids': train_ids,
                        'val_ids': val_ids,
                        'valid_locations_train': valid_locations_train,
                        'valid_locations_val': valid_locations_val}

    os.makedirs(settings.out_cache_path, exist_ok=True)
    with open(join(settings.out_cache_path, 'patch_locations.pkl'), 'wb') as handle:
        pickle.dump(contents_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def display_sample(settings):
    left_image_paths, right_image_paths, disparity_image_paths = \
            load_image_paths(settings.data_path, settings.left_img_folder,
                             settings.left_img_folder, settings.disparity_folder)
    idx = random.randint(0, len(left_image_paths))
    l_img = np.array(Image.open(left_image_paths[idx]))
    r_img = np.array(Image.open(right_image_paths[idx]))
    disp_img = np.array(Image.open(disparity_image_paths[idx]))

    show_images([l_img, r_img, disp_img], cols=2, titles=['left image',
                                                          'right image',
                                                          'disparity'])


def show_images(images, cols = 1, titles = None):
    """Display multiple images arranged as a table.

    Args:
        images (list): list of images to display as numpy arrays.
        cols (int, optional): number of columns.
        titles (list, optional): list of title strings for each image.

    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
