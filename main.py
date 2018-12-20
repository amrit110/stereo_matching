"""Main project script.

This script is used to train a TensorFlow re-implementation of
Luo, W., & Schwing, A. G. (n.d.). Efficient Deep Learning for Stereo Matching.

"""

# Imports.
import os
from os.path import join, isfile
import argparse
import random
from random import shuffle
import pickle
import logging

import matplotlib
matplotlib.use('agg')
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from lib.model import SiameseStereoMatching
from lib.dataset import Dataset
from lib.pre_process import *
from lib.utils import setup_logging

# Enable eager execution.
tf.enable_eager_execution()


# Parse args.
parser = argparse.ArgumentParser(
    description='Re-implementation of Efficient Deep Learning for Stereo Matching')
parser.add_argument('--resume', '-r', default=False, help='resume from checkpoint')
parser.add_argument('--exp-name', default='kitti_2012_run', type=str,
                    help='name of experiment')
parser.add_argument('--log-level', default='INFO', choices = ['DEBUG', 'INFO'],
                    help='log-level to use')
parser.add_argument('--batch-size', default=128, help='batch-size to use')
parser.add_argument('--dataset', default='kitti_2012', choices=['kitti_2012',
                                                                'kitti_2015'],
                    help='dataset')
parser.add_argument('--seed', default=3, help='random seed')
parser.add_argument('--patch-size', default=37, help='patch size from left image')
parser.add_argument('--disparity-range', default=201, help='disparity range')
parser.add_argument('--learning-rate', default=0.01, help='initial learning rate')
parser.add_argument('--find-patch-locations', default=False,
                    help='find and store patch locations')
parser.add_argument('--num_iterations', default=40000,
                    help='number of training iterations')
parser.add_argument('--phase', default='training', choices=['training', 'testing'],
                    help='training or testing, if testing perform inference on test set.')
parser.add_argument('--post-process', default=False,
                    help='toggle use of post-processing.')
parser.add_argument('--eval', default=False,
                    help='compute error on validation set.')

settings = parser.parse_args()


# Settings, hyper-parameters.
setattr(settings, 'data_path', join('data', settings.dataset, settings.phase))
setattr(settings, 'out_cache_path', join('cache', settings.dataset,
                                         settings.phase))
setattr(settings, 'img_height', 370)
setattr(settings, 'img_width', 1224)
setattr(settings, 'half_patch_size', (settings.patch_size // 2))
setattr(settings, 'half_range', settings.disparity_range // 2)
setattr(settings, 'num_train', 160)

if settings.dataset == 'kitti_2012':
    setattr(settings, 'left_img_folder', 'image_0')
    setattr(settings, 'right_img_folder', 'image_1')
    setattr(settings, 'disparity_folder', 'disp_noc')
    setattr(settings, 'num_val', 34)
    setattr(settings, 'num_input_channels', 1)
elif settings.dataset == 'kitti_2015':
    setattr(settings, 'left_img_folder', 'image_2')
    setattr(settings, 'right_img_folder', 'image_3')
    setattr(settings, 'disparity_folder', 'disp_noc_0')
    setattr(settings, 'num_val', 40)
    setattr(settings, 'num_input_channels', 3)


# Python logging.
LOGGER = logging.getLogger(__name__)
exp_dir = join('experiments', '{}'.format(settings.exp_name))
log_file = join(exp_dir, 'log.log')
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(join(exp_dir, 'qualitative_samples'), exist_ok=True)
setup_logging(log_path=log_file, log_level=settings.log_level, logger=LOGGER)
settings_file = join(exp_dir, 'settings.log')
with open(settings_file, 'w') as the_file:
    the_file.write(str(settings))


# Set random seed.
# NOTE: The seed affects the train/val split if patch locations data is
# created, useful for reproducing results..
random.seed(settings.seed)


# Model.
device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'
global_step = tf.Variable(0, trainable=False)
with tf.device(device):
    model = SiameseStereoMatching(settings, device, exp_dir, LOGGER, global_step)


# Optimizer
boundaries, lr_values = [24000, 32000], [settings.learning_rate,
                                         settings.learning_rate/5,
                                         settings.learning_rate/25]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, lr_values)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)


# Training/Testing
if settings.phase == 'training':
    # Find patch locations or load from cache.
    patch_locations_path = join(settings.out_cache_path, 'patch_locations.pkl')
    if settings.find_patch_locations or not isfile(patch_locations_path):
        LOGGER.info('Calculating patch locations ...')
        find_and_store_patch_locations(settings)
    with open(patch_locations_path, 'rb') as handle:
        patch_locations = pickle.load(handle)
        LOGGER.info('Patch locations loaded ...')

    # Dataset iterators.
    LOGGER.info('Initializing training and validation datasets ...')
    training_dataset = Dataset(settings, patch_locations, phase='train')
    validation_dataset = Dataset(settings, patch_locations, phase='val')

    if settings.eval:
        model.restore_model('checkpoints-40000')
        validation_dataset = Dataset(settings, patch_locations, phase='val')
        error = model.run_inference_val(validation_dataset)
        LOGGER.info('Validation 3 pixel error: {}'.format(error))
    else:
        # Training.
        LOGGER.info('Starting training ...')
        model.fit(training_dataset, validation_dataset, optimizer,
                  settings.num_iterations)
        LOGGER.info('Training done ...')
elif settings.phase == 'testing':
    model.restore_model('checkpoints-40000')
    testing_dataset = Dataset(settings, None, phase='test')
    model.run_inference_on_test(testing_dataset)
