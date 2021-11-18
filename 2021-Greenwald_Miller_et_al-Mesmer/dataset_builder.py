# Functions for building the TissueNet dataset
# This is now deprecated in favor of deepcell.datasets

# Copyright 2016-2020 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/caliban-toolbox/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import json
import warnings

import warnings
import math

import numpy as np

from skimage.measure import regionprops_table
from sklearn.model_selection import train_test_split

from deepcell_toolbox.utils import resize, tile_image

from skimage.measure import label
from skimage.morphology import remove_small_objects


def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def list_npzs_folder(npz_dir):
    """Gets all npzs in a given folder

    Args:
        npz_dir: full path to folder containing npzs

    Returns:
        list: sorted npz files
    """

    all_files = os.listdir(npz_dir)
    npz_list = [i for i in all_files if ".npz" in i]
    npz_list = sorted_nicely(npz_list)

    return npz_list


def list_folders(base_dir):
    """Lists all folders in current directory

    Args:
        base_dir: directory with folders

    Returns:
        list of folders in base_dir, empty if None
    """

    files = os.listdir(base_dir)
    folders = [file for file in files if os.path.isdir(os.path.join(base_dir, file))]

    return folders


def compute_cell_size(npz_file, method='median', by_image=True):
    """Computes the typical cell size from a stack of labeled data

    Args:
        npz_file: Paired X and y data
        method: one of (mean, median) used to compute the cell size
        by_image: if true, cell size is reported for each image in npz_file. Otherwise,
            the cell size across the entire npz is returned

    Returns:
        list: list of typical cell size in NPZ. If no cells, returns None.

    Raises: ValueError if invalid method supplied
    Raises: ValueError if data does have len(shape) of 4
    """

    valid_methods = {'median', 'mean'}
    method = str(method).lower()
    if method not in valid_methods:
        raise ValueError('Invalid method supplied: got {}, '
                         'method must be one of {}'.format(method, valid_methods))

    # initialize variables
    cell_sizes = []
    labels = npz_file['y']

    if len(labels.shape) != 4:
        raise ValueError('Labeled data must be 4D')

    for i in range(labels.shape[0]):
        current_label = labels[i, :, :, 0]

        # check to make sure array contains cells
        if len(np.unique(current_label)) > 1:
            area = regionprops_table(current_label.astype('int'), properties=['area'])['area']
            cell_sizes.append(area)

    # if all images were empty, return NA
    if cell_sizes == []:
        return None

    # compute for each list corresponding to each image
    if by_image:
        average_cell_sizes = []
        for size_list in cell_sizes:
            if method == 'mean':
                average_cell_sizes.append(np.mean(size_list))
            elif method == 'median':
                average_cell_sizes.append(np.median(size_list))

    # compute across all lists from all images
    else:
        all_cell_sizes = np.concatenate(cell_sizes)
        if method == 'mean':
            average_cell_sizes = np.mean(all_cell_sizes)
        elif method == 'median':
            average_cell_sizes = np.median(all_cell_sizes)
        else:
            raise ValueError('Invalid method supplied')

    return average_cell_sizes


def reshape_training_data(X_data, y_data, resize_ratio, final_size, stride_ratio=1, tolerance=1.5):
    """Takes a stack of X and y data and reshapes and crops them to match output dimensions

    Args:
        X_data: 4D numpy array of image data
        y_data: 4D numpy array of labeled data
        resize_ratio: resize ratio for the images
        final_size: the desired shape of the output image
        stride_ratio: amount of overlap between crops (1 is no overlap, 0.5 is half crop size)
        tolerance: ratio that determines when resizing occurs

    Returns:
        reshaped_X, reshaped_y: resized and cropped version of input images

    Raises:
        ValueError: If image data is not 4D
    """

    if len(X_data.shape) != 4:
        raise ValueError('Image data must be 4D')

    # resize if needed
    if resize_ratio > tolerance or resize_ratio < (1 / tolerance):
        new_shape = (int(X_data.shape[1] * resize_ratio),
                     int(X_data.shape[2] * resize_ratio))

        X_data = resize(data=X_data, shape=new_shape)
        y_data = resize(data=y_data, shape=new_shape, labeled_image=True)

    # crop if needed
    if X_data.shape[1:3] != final_size:
        # pad image so that crops divide evenly
        X_data = pad_image_stack(images=X_data, crop_size=final_size)
        y_data = pad_image_stack(images=y_data, crop_size=final_size)

        # create x and y crops
        X_data, _ = tile_image(image=X_data, model_input_shape=final_size,
                               stride_ratio=stride_ratio)
        y_data, _ = tile_image(image=y_data, model_input_shape=final_size,
                               stride_ratio=stride_ratio)
    return X_data, y_data


def train_val_test_split(X_data, y_data, data_split=(0.8, 0.1, 0.1), seed=None):
    """Randomly splits supplied data into specified sizes for model assessment

    Args:
        X_data: array of X data
        y_data: array of y_data
        data_split: tuple specifying the fraction of the dataset for train/val/test
        seed: random seed for reproducible splits

    Returns:
        list of X and y data split appropriately. If dataset is too small for all splits,
            returns None for remaining splits

    Raises:
        ValueError: if ratios do not sum to 1
        ValueError: if any of the splits are 0
        ValueError: If length of X and y data is not equal
    """

    total = np.round(np.sum(data_split), decimals=2)
    if total != 1:
        raise ValueError('Data splits must sum to 1, supplied splits sum to {}'.format(total))

    if 0 in data_split:
        raise ValueError('All splits must be non-zero')

    if X_data.shape[0] != y_data.shape[0]:
        raise ValueError('Supplied X and y data do not have the same '
                         'length over batches dimension. '
                         'X.shape: {}, y.shape: {}'.format(X_data.shape, y_data.shape))

    train_ratio, val_ratio, test_ratio = data_split

    # 1 image, train split only
    if X_data.shape[0] == 1:
        warnings.warn('Only one image in current NPZ, returning training split only')
        return X_data, y_data, None, None, None, None

    # 2 images, train and val split only
    if X_data.shape[0] == 2:
        warnings.warn('Only two images in current NPZ, returning training and val split only')
        return X_data[:1, ...], y_data[:1, ...], X_data[1:, ...], y_data[1:, ...], None, None

    # compute fraction not in train
    val_remainder_ratio = np.round(1 - train_ratio, decimals=2)
    val_remainder_count = X_data.shape[0] * val_remainder_ratio

    # not enough data for val split, put minimum (1) in each split
    if val_remainder_count < 1:
        warnings.warn('Not enough data in current NPZ for specified data split.'
                      'Returning modified data split')
        X_train, X_remainder, y_train, y_remainder = train_test_split(X_data, y_data,
                                                                      test_size=2,
                                                                      random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_remainder, y_remainder,
                                                        test_size=1,
                                                        random_state=seed)
        return X_train, y_train, X_val, y_val, X_test, y_test

    # split dataset into train and remainder
    X_train, X_remainder, y_train, y_remainder = train_test_split(X_data, y_data,
                                                                  test_size=val_remainder_ratio,
                                                                  random_state=seed)

    # compute fraction of remainder that is test
    test_remainder_ratio = np.round(test_ratio / (val_ratio + test_ratio), decimals=2)
    test_remainder_count = X_remainder.shape[0] * test_remainder_ratio

    # not enough data in remainder for test split, put minimum (1) in test split from train split
    if test_remainder_count < 1:
        warnings.warn('Not enough data in current NPZ for specified data split.'
                      'Returning modified data split')
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=1,
                                                            random_state=seed)
        X_val, y_val = X_remainder, y_remainder

        return X_train, y_train, X_val, y_val, X_test, y_test

    # split remainder into val and test
    X_val, X_test, y_val, y_test = train_test_split(X_remainder, y_remainder,
                                                    test_size=test_remainder_ratio,
                                                    random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


class DatasetBuilder(object):
    """Class to build a dataset from annotated data

    Args:
        dataset_path: path to dataset. Within the dataset, each unique experiment
            has its own folder with a dedicated metadata file
    """
    def __init__(self, dataset_path):

        self._validate_dataset(dataset_path)

        experiment_folders = list_folders(dataset_path)
        self.dataset_path = dataset_path
        self.experiment_folders = experiment_folders

        self.all_tissues = None
        self.all_platforms = None

        # dicts to hold aggregated data
        self.train_dict = {}
        self.val_dict = {}
        self.test_dict = {}

        # parameters for splitting the data
        self.data_split = None
        self.seed = None

    def _validate_dataset(self, dataset_path):
        """Check to make sure that supplied dataset is formatted appropriately

        Args:
            dataset_path: path to dataset

        Raises:
            ValueError: If dataset_path doesn't exist
            ValueError: If dataset_path doesn't contain any folders
            ValueError: If dataset_path has any folders without an NPZ file
            ValueError: If dataset_path has any folders without a metadata file
        """

        if not os.path.isdir(dataset_path):
            raise ValueError('Invalid dataset_path, must be a directory')
        experiment_folders = list_folders(dataset_path)

        if experiment_folders == []:
            raise ValueError('dataset_path must include at least one folder')

        for folder in experiment_folders:
            if not os.path.exists(os.path.join(dataset_path, folder, 'metadata.json')):
                raise ValueError('No metadata file found in {}'.format(folder))
            npz_files = list_npzs_folder(os.path.join(dataset_path, folder))

            if len(npz_files) == 0:
                raise ValueError('No NPZ files found in {}'.format(folder))

    def _get_metadata(self, experiment_folder):
        """Get the metadata associated with a specific experiment

        Args:
            experiment_folder: folder to get metadata from

        Returns:
            dictionary containing relevant metadata"""

        metadata_file = os.path.join(self.dataset_path, experiment_folder, 'metadata.json')
        with open(metadata_file) as f:
            metadata = json.load(f)

        return metadata

    def _identify_tissue_and_platform_types(self):
        """Identify all of the unique tissues and platforms in the dataset"""

        tissues = []
        platforms = []
        for folder in self.experiment_folders:
            metadata = self._get_metadata(experiment_folder=folder)

            tissues.append(metadata['tissue_specific'])
            platforms.append(metadata['platform'])

        self.all_tissues = np.array(tissues)
        self.all_platforms = np.array(platforms)

    def _load_experiment(self, experiment_path):
        """Load the NPZ files present in a single experiment folder

        Args:
            experiment_path: the full path to a folder of NPZ files and metadata file

        Returns:
            tuple of X and y data from all NPZ files in the experiment
            tissue: the tissue type of this experiment
            platform: the platform type of this experiment
        """

        X_list = []
        y_list = []

        # get all NPZ files present in current experiment directory
        npz_files = list_npzs_folder(experiment_path)
        for file in npz_files:
            npz_path = os.path.join(experiment_path, file)
            training_data = np.load(npz_path)

            X = training_data['X']
            y = training_data['y']

            X_list.append(X)
            y_list.append(y)

        # get associated metadata
        metadata = self._get_metadata(experiment_folder=experiment_path)

        # combine all NPZ files together
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        if np.issubdtype(y.dtype, np.floating):
            warnings.warn('Converting float labels to integers')
            y = y.astype('int64')

        tissue = metadata['tissue_specific']
        platform = metadata['platform']

        return X, y, tissue, platform

    def _load_all_experiments(self, data_split, seed):
        """Loads all experiment data from experiment folder to enable dataset building

        Args:
            data_split: tuple specifying the fraction of the dataset for train/val/test
            seed: seed for reproducible splitting of dataset

        Raises:
            ValueError: If any of the NPZ files have different non-batch dimensions
        """
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []
        tissue_list_train, tissue_list_val, tissue_list_test = [], [], []
        platform_list_train, platform_list_val, platform_list_test = [], [], []

        # loop through all experiments
        for folder in self.experiment_folders:
            # Get all NPZ files from each experiment
            folder_path = os.path.join(self.dataset_path, folder)
            X, y, tissue, platform = self._load_experiment(folder_path)

            # split data according to specified ratios
            X_train_batch, y_train_batch, X_val_batch, y_val_batch, X_test_batch, y_test_batch = \
                train_val_test_split(X_data=X, y_data=y, data_split=data_split, seed=seed)

            # construct list for each split
            tissue_list_train_batch = [tissue] * X_train_batch.shape[0]
            platform_list_train_batch = [platform] * X_train_batch.shape[0]
            X_train.append(X_train_batch)
            y_train.append(y_train_batch)
            tissue_list_train.append(tissue_list_train_batch)
            platform_list_train.append(platform_list_train_batch)

            if X_val_batch is not None:
                tissue_list_val_batch = [tissue] * X_val_batch.shape[0]
                platform_list_val_batch = [platform] * X_val_batch.shape[0]
                X_val.append(X_val_batch)
                y_val.append(y_val_batch)
                tissue_list_val.append(tissue_list_val_batch)
                platform_list_val.append(platform_list_val_batch)

            if X_test_batch is not None:
                tissue_list_test_batch = [tissue] * X_test_batch.shape[0]
                platform_list_test_batch = [platform] * X_test_batch.shape[0]
                X_test.append(X_test_batch)
                y_test.append(y_test_batch)
                tissue_list_test.append(tissue_list_test_batch)
                platform_list_test.append(platform_list_test_batch)

        # make sure that all data has same shape
        first_shape = X_train[0].shape
        for i in range(1, len(X_train)):
            current_shape = X_train[i].shape
            if first_shape[1:] != current_shape[1:]:
                raise ValueError('Found mismatching dimensions between '
                                 'first NPZ and npz at position {}. '
                                 'Shapes of {}, {}'.format(i, first_shape, current_shape))

        # concatenate lists together
        X_train = np.concatenate(X_train, axis=0)
        X_val = np.concatenate(X_val, axis=0)
        X_test = np.concatenate(X_test, axis=0)

        y_train = np.concatenate(y_train, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        tissue_list_train = np.concatenate(tissue_list_train, axis=0)
        tissue_list_val = np.concatenate(tissue_list_val, axis=0)
        tissue_list_test = np.concatenate(tissue_list_test, axis=0)

        platform_list_train = np.concatenate(platform_list_train, axis=0)
        platform_list_val = np.concatenate(platform_list_val, axis=0)
        platform_list_test = np.concatenate(platform_list_test, axis=0)

        # create combined dicts
        train_dict = {'X': X_train, 'y': y_train, 'tissue_list': tissue_list_train,
                      'platform_list': platform_list_train}

        val_dict = {'X': X_val, 'y': y_val, 'tissue_list': tissue_list_val,
                    'platform_list': platform_list_val}

        test_dict = {'X': X_test, 'y': y_test, 'tissue_list': tissue_list_test,
                     'platform_list': platform_list_test}

        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict
        self.data_split = data_split
        self.seed = seed

    def _subset_data_dict(self, data_dict, tissues, platforms):
        """Subsets a dictionary to only include from the specified tissues and platforms

        Args:
            data_dict: dictionary to subset from
            tissues: list of tissues to include
            platforms: list of platforms to include

        Returns:
            subset_dict: dictionary containing examples desired data

        Raises:
            ValueError: If no matching data for tissue/platform combination
        """
        X, y = data_dict['X'], data_dict['y']
        tissue_list, platform_list = data_dict['tissue_list'], data_dict['platform_list']

        # Identify locations with the correct categories types
        tissue_idx = np.isin(tissue_list, tissues)
        platform_idx = np.isin(platform_list, platforms)

        # get indices which meet both criteria
        combined_idx = tissue_idx * platform_idx

        # check that there is data which meets requirements
        if np.sum(combined_idx) == 0:
            raise ValueError('No matching data for specified parameters')

        X, y = X[combined_idx], y[combined_idx]
        tissue_list = tissue_list[combined_idx]
        platform_list = platform_list[combined_idx]

        subset_dict = {'X': X, 'y': y, 'tissue_list': tissue_list,
                       'platform_list': platform_list}
        return subset_dict

    def _reshape_dict(self, data_dict, resize=False, output_shape=(512, 512), resize_target=400,
                      resize_tolerance=1.5):
        """Takes a dictionary of training data and reshapes it to appropriate size

        data_dict: dictionary of training data
        resize: flag to control resizing of the data.
            Valid arguments:
                    - False. No resizing
                    - int/float: resizes by given ratio for all images
                    - by_tissue. Resizes by median cell size within each tissue type
                    - by_image. Resizes by median cell size within each image
        output_shape: output shape for image data
        resize_target: desired median cell size after resizing
        resize_tolerance: sets maximum allowable ratio between resize_target and
            median cell size before resizing occurs
        """
        X, y = data_dict['X'], data_dict['y']
        tissue_list = data_dict['tissue_list']
        platform_list = data_dict['platform_list']

        if not resize:
            # no resizing
            X_new, y_new = reshape_training_data(X_data=X, y_data=y, resize_ratio=1,
                                                 final_size=output_shape, stride_ratio=1)

            # to preserve category labels, we need to figure out how much the array grew by
            multiplier = int(X_new.shape[0] / X.shape[0])

            # then we duplicate the labels in place to match expanded array size
            tissue_list_new = np.repeat(tissue_list, multiplier)
            platform_list_new = np.repeat(platform_list, multiplier)

        elif isinstance(resize, (float, int)):
            # resized based on supplied value
            X_new, y_new = reshape_training_data(X_data=X, y_data=y, resize_ratio=resize,
                                                 final_size=output_shape, stride_ratio=1,
                                                 tolerance=resize_tolerance)

            # to preserve category labels, we need to figure out how much the array grew by
            multiplier = int(X_new.shape[0] / X.shape[0])

            # then we duplicate the labels in place to match expanded array size
            tissue_list_new = np.repeat(tissue_list, multiplier)
            platform_list_new = np.repeat(platform_list, multiplier)
        else:
            X_new, y_new, tissue_list_new, platform_list_new = [], [], [], []

            if resize == 'by_tissue':
                batch_ids = np.unique(tissue_list)
            elif resize == 'by_image':
                batch_ids = np.arange(0, X.shape[0])
            else:
                raise ValueError('Invalid `resize` value: {}'.format(resize))

            # loop over each batch
            for batch_id in batch_ids:

                # get tissue types that match current tissue type
                if isinstance(batch_id, str):
                    batch_idx = np.isin(tissue_list, batch_id)

                # get boolean index for current image
                else:
                    batch_idx = np.arange(X.shape[0]) == batch_id

                X_batch, y_batch = X[batch_idx], y[batch_idx]
                tissue_list_batch = tissue_list[batch_idx]
                platform_list_batch = platform_list[batch_idx]

                # compute appropriate resize ratio
                median_cell_size = compute_cell_size({'X': X_batch, 'y': y_batch}, by_image=False)

                # check for empty images
                if median_cell_size is not None:
                    resize_ratio = np.sqrt(resize_target / median_cell_size)
                else:
                    resize_ratio = 1

                # resize the data
                X_batch_resized, y_batch_resized = reshape_training_data(
                    X_data=X_batch, y_data=y_batch,
                    resize_ratio=resize_ratio, final_size=output_shape,
                    tolerance=resize_tolerance)

                # to preserve category labels, we need to figure out how much the array grew by
                multiplier = int(X_batch_resized.shape[0] / X_batch.shape[0])

                # then we duplicate the labels in place to match expanded array size
                tissue_list_batch = np.repeat(tissue_list_batch, multiplier)
                platform_list_batch = np.repeat(platform_list_batch, multiplier)

                # add each batch onto main list
                X_new.append(X_batch_resized)
                y_new.append(y_batch_resized)
                tissue_list_new.append(tissue_list_batch)
                platform_list_new.append(platform_list_batch)

            X_new = np.concatenate(X_new, axis=0)
            y_new = np.concatenate(y_new, axis=0)
            tissue_list_new = np.concatenate(tissue_list_new, axis=0)
            platform_list_new = np.concatenate(platform_list_new, axis=0)

        return {'X': X_new, 'y': y_new, 'tissue_list': tissue_list_new,
                'platform_list': platform_list_new}

    def _clean_labels(self, data_dict, relabel=False, small_object_threshold=0,
                      min_objects=0):
        """Cleans labels prior to creating final dict

        Args:
            data_dict: dictionary of training data
            relabel: if True, relabels the image with new labels
            small_object_threshold: threshold for removing small objects
            min_objects: minimum number of objects per image

        Returns:
            cleaned_dict: dictionary with cleaned labels
        """
        X, y = data_dict['X'], data_dict['y']
        tissue_list = data_dict['tissue_list']
        platform_list = data_dict['platform_list']
        keep_idx = np.repeat(True, y.shape[0])
        cleaned_y = np.zeros_like(y)

        # TODO: remove once data QC happens in main toolbox pipeline
        for i in range(y.shape[0]):
            y_current = y[i, ..., 0]
            if relabel:
                y_current = label(y_current)

            y_current = remove_small_objects(y_current, min_size=small_object_threshold)

            unique_objects = len(np.unique(y_current)) - 1
            if unique_objects < min_objects:
                keep_idx[i] = False

            cleaned_y[i, ..., 0] = y_current

        # subset all dict members to include only relevant images
        cleaned_y = cleaned_y[keep_idx]
        cleaned_X = X[keep_idx]
        cleaned_tissue = tissue_list[keep_idx]
        cleaned_platform = platform_list[keep_idx]

        cleaned_dict = {'X': cleaned_X, 'y': cleaned_y, 'tissue_list': cleaned_tissue,
                        'platform_list': cleaned_platform}

        return cleaned_dict

    def _balance_dict(self, data_dict, seed, category):
        """Balance a dictionary of training data so that each category is equally represented

        Args:
            data_dict: dictionary of training data
            seed: seed for random duplication of less-represented classes
            category: name of the key in the dictionary to use for balancing

        Returns:
            dict: training data that has been balanced
        """

        np.random.seed(seed)
        category_list = data_dict[category]

        unique_categories, unique_counts = np.unique(category_list, return_counts=True)
        max_counts = np.max(unique_counts)

        # original variables
        X_unbalanced, y_unbalanced = data_dict['X'], data_dict['y']
        tissue_unbalanced = np.array(data_dict['tissue_list'])
        platform_unbalanced = np.array(data_dict['platform_list'])

        # create list to hold balanced versions
        X_balanced, y_balanced, tissue_balanced, platform_balanced = [], [], [], []
        for category in unique_categories:
            cat_idx = category == category_list
            X_cat, y_cat = X_unbalanced[cat_idx], y_unbalanced[cat_idx]
            tissue_cat, platform_cat = tissue_unbalanced[cat_idx], platform_unbalanced[cat_idx]

            category_counts = X_cat.shape[0]
            if category_counts == max_counts:
                # we don't need to balance, as this category already has max number of examples
                X_balanced.append(X_cat)
                y_balanced.append(y_cat)
                tissue_balanced.append(tissue_cat)
                platform_balanced.append(platform_cat)
            else:
                # randomly select max_counts number of indices to upsample data
                balance_idx = np.random.choice(range(category_counts), size=max_counts,
                                               replace=True)

                # index into each array using random index to generate randomly upsampled version
                X_balanced.append(X_cat[balance_idx])
                y_balanced.append(y_cat[balance_idx])
                tissue_balanced.append(tissue_cat[balance_idx])
                platform_balanced.append(platform_cat[balance_idx])

        # combine balanced versions of each category into single array
        X_balanced = np.concatenate(X_balanced, axis=0)
        y_balanced = np.concatenate(y_balanced, axis=0)
        tissue_balanced = np.concatenate(tissue_balanced, axis=0)
        platform_balanced = np.concatenate(platform_balanced, axis=0)

        return {'X': X_balanced, 'y': y_balanced, 'tissue_list': tissue_balanced,
                'platform_list': platform_balanced}

    def _validate_categories(self, category_list, supplied_categories):
        """Check that an appropriate subset of a list of categories was supplied

        Args:
            category_list: list of all categories
            supplied_categories: specified categories provided by user. Must be either
                - a list containing the desired category members
                - a string of a single category name
                - a string of 'all', in which case all will be used

        Returns:
            list: a properly formatted sub_category list

        Raises:
            ValueError: if invalid supplied_categories argument
            """
        if isinstance(supplied_categories, list):
            for cat in supplied_categories:
                if cat not in category_list:
                    raise ValueError('{} is not one of {}'.format(cat, category_list))
            return supplied_categories
        elif supplied_categories == 'all':
            return category_list
        elif supplied_categories in category_list:
            return [supplied_categories]
        else:
            raise ValueError(
                'Specified categories should be "all", one of {}, or a list '
                'of acceptable tissue types'.format(category_list))

    def _validate_output_shape(self, output_shape):
        """Check that appropriate values were provided for output_shape

        Args:
            output_shape: output_shape supplied by the user

        Returns:
            list: a properly formatted output_shape

        Raises:
            ValueError: If invalid output_shape provided
        """
        if not isinstance(output_shape, (list, tuple)):
            raise ValueError('output_shape must be either a list of tuples or a tuple')

        if len(output_shape) == 2:
            for val in output_shape:
                if not isinstance(val, int):
                    raise ValueError('A list of length two was supplied, but not all '
                                     'list items were ints, got {}'.format(val))
            # convert to list with same shape for each split
            output_shape = [output_shape, output_shape, output_shape]
            return output_shape
        elif len(output_shape) == 3:
            for sub_shape in output_shape:
                if not len(sub_shape) == 2:
                    raise ValueError('A list of length three was supplied, bu not all '
                                     'of the sublists had len 2, got {}'.format(sub_shape))
                for val in sub_shape:
                    if not isinstance(val, int):
                        raise ValueError('A list of lists was supplied, but not all '
                                         'sub_list items were ints, got {}'.format(val))

            return output_shape
        else:
            raise ValueError('output_shape must be a list of length 2 '
                             'or length 3, got {}'.format(output_shape))

    def build_dataset(self, tissues='all', platforms='all', output_shape=(512, 512), resize=False,
                      data_split=(0.8, 0.1, 0.1), seed=0, balance=False, **kwargs):
        """Construct a dataset for model training and evaluation

        Args:
            tissues: which tissues to include. Must be either a list of tissue types,
                a single tissue type, or 'all'
            platforms: which platforms to include. Must be either a list of platform types,
                a single platform type, or 'all'
            output_shape: output shape for dataset. Either a single tuple, in which case
                train/va/test will all have same size, or a list of three tuples
            resize: flag to control resizing the input data.
                Valid arguments:
                    - False. No resizing
                    - float/int: Resizes all images by supplied value
                    - by_tissue. Resizes by median cell size within each tissue type
                    - by_image. Resizes by median cell size within each image
            data_split: tuple specifying the fraction of the dataset for train/val/test
            seed: seed for reproducible splitting of dataset
            balance: if true, randomly duplicate less-represented tissue types
                in train and val splits so that there are the same number of images of each type
            **kwargs: other arguments to be passed to helper functions

        Returns:
            list of dicts containing the split dataset

        Raises:
            ValueError: If invalid resize parameter supplied
        """
        if self.all_tissues is None:
            self._identify_tissue_and_platform_types()

        # validate inputs
        tissues = self._validate_categories(category_list=self.all_tissues,
                                            supplied_categories=tissues)
        platforms = self._validate_categories(category_list=self.all_platforms,
                                              supplied_categories=platforms)

        valid_resize = ['by_tissue', 'by_image']
        if resize in valid_resize or not resize:
            pass
        elif isinstance(resize, (float, int)):
            if resize <= 0:
                raise ValueError('Resize values must be greater than 0')
        else:
            raise ValueError('resize must be one of {}, or an integer value'.format(valid_resize))

        output_shape = self._validate_output_shape(output_shape=output_shape)

        # if any of the split parameters are different we need to reload the dataset
        if self.seed != seed or self.data_split != data_split:
            self._load_all_experiments(data_split=data_split, seed=seed)

        dicts = [self.train_dict, self.val_dict, self.test_dict]
        # process each dict
        for idx, current_dict in enumerate(dicts):
            # subset dict to include only relevant tissues and platforms
            current_dict = self._subset_data_dict(data_dict=current_dict, tissues=tissues,
                                                  platforms=platforms)
            current_shape = output_shape[idx]

            # if necessary, reshape and resize data to be of correct output size
            if current_dict['X'].shape[1:3] != current_shape or resize is not False:
                resize_target = kwargs.get('resize_target', 400)
                resize_tolerance = kwargs.get('resize_tolerance', 1.5)
                current_dict = self._reshape_dict(data_dict=current_dict, resize=resize,
                                                  output_shape=current_shape,
                                                  resize_target=resize_target,
                                                  resize_tolerance=resize_tolerance)

            # clean labels
            relabel = kwargs.get('relabel', False)
            small_object_threshold = kwargs.get('small_object_threshold', 0)
            min_objects = kwargs.get('min_objects', 0)
            current_dict = self._clean_labels(data_dict=current_dict, relabel=relabel,
                                              small_object_threshold=small_object_threshold,
                                              min_objects=min_objects)

            # don't balance test split
            if balance and idx != 2:
                current_dict = self._balance_dict(current_dict, seed=seed, category='tissue_list')

            dicts[idx] = current_dict
        return dicts

    def summarize_dataset(self):
        """Computes summary statistics for the images in the dataset

        Returns:
            dict of cell counts and image counts by tissue
            dict of cell counts and image counts by platform
        """
        all_y = np.concatenate((self.train_dict['y'],
                                self.val_dict['y'],
                                self.test_dict['y']),
                               axis=0)
        all_tissue = np.concatenate((self.train_dict['tissue_list'],
                                     self.val_dict['tissue_list'],
                                     self.test_dict['tissue_list']),
                                    axis=0)

        all_platform = np.concatenate((self.train_dict['platform_list'],
                                       self.val_dict['platform_list'],
                                       self.test_dict['platform_list']),
                                      axis=0)
        all_counts = np.zeros(all_y.shape[0])
        for i in range(all_y.shape[0]):
            unique_counts = len(np.unique(all_y[i, ..., 0])) - 1
            all_counts[i] = unique_counts

        tissue_dict = {}
        for tissue in np.unique(all_tissue):
            tissue_idx = np.isin(all_tissue, tissue)
            tissue_counts = np.sum(all_counts[tissue_idx])
            tissue_unique = np.sum(tissue_idx)
            tissue_dict[tissue] = {'cell_num': tissue_counts,
                                   'image_num': tissue_unique}

        platform_dict = {}
        for platform in np.unique(all_platform):
            platform_idx = np.isin(all_platform, platform)
            platform_counts = np.sum(all_counts[platform_idx])
            platform_unique = np.sum(platform_idx)
            platform_dict[platform] = {'cell_num': platform_counts,
                                       'image_num': platform_unique}

        return tissue_dict, platform_dict

# example use case
data_dir = 'path_to_tissuenet/'

db = DatasetBuilder(data_dir)
train, val, test = db.build_dataset(tissues='all', platforms='all',
                                    output_shape=[(512, 512), (256, 256), (512, 512)],
                                    resize=None, relabel=True, min_objects=20,
                                    small_object_threshold=15, balance=False, seed=1)

val_small = db._reshape_dict(data_dict=val, resize=0.707, output_shape=(256, 256),
                             resize_tolerance=1)
val_large = db._reshape_dict(data_dict=val, resize=1.41, output_shape=(256, 256),
                             resize_tolerance=1)


def subset_resized_dict(original_dict, large_dict):
    img_num = original_dict['X'].shape[0]
    large_num = large_dict['X'].shape[0]
    np.random.seed(0)
    indices = np.random.choice(range(large_num), img_num, replace=False)

    for key in large_dict:
        large_entry = large_dict[key]
        resized_entry = large_entry[indices]
        large_dict[key] = resized_entry

    return large_dict


def combine_dicts(dict_list):
    keys = dict_list[0].keys()
    combined = {}
    for key in keys:
        all_values = [current_dict[key] for current_dict in dict_list]
        new_value = np.concatenate(all_values, axis=0)
        combined[key] = new_value
    return combined


def reduce_dtypes(input_dict):
    X_data = input_dict['X']
    X_max = np.max(X_data, axis=(1, 2, 3))
    X_max = np.expand_dims(np.expand_dims(np.expand_dims(X_max, axis=-1), axis=-1), axis=-1)
    X_data = X_data / X_max
    X_data = X_data.astype('float16')

    y_data = input_dict['y']
    y_data = y_data.astype('int16')

    input_dict['X'] = X_data
    input_dict['y'] = y_data

    return input_dict


val_large = subset_resized_dict(val, val_large)
val_combined = combine_dicts((val, val_small, val_large))
val_combined = db._clean_labels(data_dict=val_combined, min_objects=20)


train = reduce_dtypes(train)
val = reduce_dtypes(val_combined)
test = reduce_dtypes(test)

prefix = '20211029'
np.savez_compressed(os.path.join(data_dir, prefix + '_train_512x512.npz'), **train)
np.savez_compressed(os.path.join(data_dir, prefix + '_val_256x256.npz'), **val)
np.savez_compressed(os.path.join(data_dir, prefix + '_test_512x512.npz'), **test)