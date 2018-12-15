"""Visualize KITTI stereo and disparity images. Pre-process to create dataset
to sample mini-batches for training."""


# Imports.
import os
import random
import glob
import pickle
from os.path import join
from random import shuffle

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Dataset related globals.
DATASET = 'kitti_2015' # options are ['kitti_2012', 'kitti_2015']
PHASE = 'training'
DATA_PATH = join('data', DATASET, PHASE)
OUT_CACHE_PATH = join('cache', DATASET, PHASE)

if DATASET == 'kitti_2012':
    LEFT_IMG_FOLDER = 'image_0'
    RIGHT_IMG_FOLDER = 'image_1'
    DISPARITY_FOLDER = 'disp_noc'
    N_VAL = 34
elif DATASET == 'kitti_2015':
    LEFT_IMG_FOLDER = 'image_2'
    RIGHT_IMG_FOLDER = 'image_3'
    DISPARITY_FOLDER = 'disp_noc_1'
    N_VAL = 40

# Pre-processing related globals.
IMG_HEIGHT = 370
IMG_WIDTH = 1224
PATCH_SIZE = 37
HALF_PATCH_SIZE = (PATCH_SIZE // 2)
DISPARITY_RANGE = 201
HALF_RANGE = DISPARITY_RANGE // 2
N_TRAIN = 160

# Fix random seed, so train/val split remains same.
random.seed(5)


def load_image_paths():
    left_image_paths = sorted(glob.glob(join(DATA_PATH, LEFT_IMG_FOLDER, '*10.png')))
    right_image_paths = sorted(glob.glob(join(DATA_PATH, RIGHT_IMG_FOLDER, '*10.png')))
    disparity_image_paths = sorted(glob.glob(join(DATA_PATH, DISPARITY_FOLDER, '*10.png')))

    return left_image_paths, right_image_paths, disparity_image_paths


def _is_valid_location(sample_locations, img_width, img_height):
    left_center_x, left_center_y, right_center_x, right_center_y = sample_locations
    is_valid_loc_left = ((left_center_x + HALF_PATCH_SIZE + 1) <= img_width) and \
            ((left_center_x - HALF_PATCH_SIZE) >= 0) and \
            ((left_center_y + HALF_PATCH_SIZE + 1) <= img_height) and \
            ((left_center_y - HALF_PATCH_SIZE) >= 0)
    is_valid_loc_right = ((right_center_x - HALF_RANGE - HALF_PATCH_SIZE) >= 0) and \
            ((right_center_x + HALF_RANGE + HALF_PATCH_SIZE + 1) <= img_width) and \
            ((right_center_y - HALF_PATCH_SIZE) >= 0) and \
            ((right_center_y + HALF_PATCH_SIZE + 1) <= img_height)

    return (is_valid_loc_left and is_valid_loc_right)


def compute_valid_locations(disparity_image_paths, sample_ids):
    num_samples = len(sample_ids)
    num_valid_locations = np.zeros(num_samples)

    for i, idx in enumerate(sample_ids):
        disp_img = np.array(Image.open(disparity_image_paths[idx])).astype('float64')
        # We want images of same size for efficient loading.
        disp_img = disp_img[0:IMG_HEIGHT, 0:IMG_WIDTH]
        disp_img /= 256
        num_valid_locations[i] = (disp_img != 0).sum()

    num_valid_locations = int(num_valid_locations.sum())
    valid_locations = np.zeros((num_valid_locations, 4))
    valid_count = 0

    for i, idx in enumerate(sample_ids):
        disp_img = np.array(Image.open(disparity_image_paths[idx])).astype('float64')
        # We want images of same size for efficient loading.
        disp_img = disp_img[0:IMG_HEIGHT, 0:IMG_WIDTH]
        disp_img /= 256
        row_locs, col_locs = np.where(disp_img != 0)
        img_height, img_width = disp_img.shape

        for j, row_loc in enumerate(row_locs):
            left_center_x = col_locs[j]
            left_center_y = row_loc
            right_center_x = int(round(col_locs[j] - disp_img[left_center_y,
                                                              left_center_x]))
            right_center_y = left_center_y # Stereo pair is assumed to be rectified.

            sample_locations = (left_center_x, left_center_y, right_center_x, right_center_y)
            if _is_valid_location(sample_locations, img_width, img_height):
                location_info = np.array([idx, left_center_x,
                                          left_center_y,
                                          right_center_x])
                valid_locations[valid_count, :] = location_info
                valid_count += 1

    valid_locations = valid_locations[0:valid_count, :]
    print(valid_locations.shape)

    return valid_locations


def find_patch_locations():
    left_image_paths, right_image_paths, disparity_image_paths = load_image_paths()
    sample_indices = list(range(len(left_image_paths)))
    shuffle(sample_indices)
    train_ids, val_ids = sample_indices[0:N_TRAIN], sample_indices[N_TRAIN:]

    # Training set.
    valid_locations_train = compute_valid_locations(disparity_image_paths, train_ids)
    # Validation set.
    valid_locations_val = compute_valid_locations(disparity_image_paths, val_ids)

    # Save to disk
    contents_to_save = {'sample_indices': sample_indices,
                        'train_ids': train_ids,
                        'val_ids': val_ids,
                        'valid_locations_train': valid_locations_train,
                        'valid_locations_val': valid_locations_val}

    os.makedirs(OUT_CACHE_PATH, exist_ok=True)
    with open(join(OUT_CACHE_PATH, 'patch_locations.pkl'), 'wb') as handle:
        pickle.dump(contents_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Patch locations cache file saved.')

def display_sample():
    left_image_paths, right_image_paths, disparity_image_paths = load_image_paths()

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


if __name__ == '__main__':
    print('Finding patch locations ...')
    find_patch_locations()
    # display_sample()
