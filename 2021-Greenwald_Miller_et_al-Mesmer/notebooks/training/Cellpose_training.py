"""Save TissueNet as individulal TIFF files.

These files are then fed into cellpose for training.

To train the cellpose model:

CUDA_VISIBLE_DEVICES=3 python -m cellpose --train \
--dir /deepcell_data/users/willgraf/mesmer_retrain/tissue_net/seed1/train/ \
--test_dir /deepcell_data/users/willgraf/mesmer_retrain/tissue_net/seed1/val/ \
--pretrained_model None \
--img_filter _img \
--mask_filter _masks \
--chan 2 --chan2 1 \
--use_gpu

To run the newly trained cellpose model:

CUDA_VISIBLE_DEVICES=3 python -m cellpose \
--dir /deepcell_data/users/willgraf/mesmer_retrain/tissue_net/seed1/test_run/ \
--pretrained_model /deepcell_data/users/willgraf/mesmer_retrain/tissue_net/seed1/train/models/cellpose_residual_on_style_on_concatenation_off_train_2021_04_26_11_35_14.114698 \
--chan 2 --chan2 1 \
--diameter 23. --save_tif --use_gpu


To run the pretrained version of the CellPose model:

python -m cellpose
    --dir /deepcell_data/users/willgraf/cellpose/test_split_1_channels_first
    --pretrained_model cyto
    --chan 0 --chan2 1
    --diameter 0.
    --save_tif --use_gpu
"""

import os
import numpy as np
import tifffile

SEED = 1

NPZ_NAME = '20201018_multiplex_seed_{}'.format(SEED)
EXP_NAME = '20200824_hyper_parameter'
MODEL_NAME = '{}_cellpose'.format(NPZ_NAME)

ROOT_DIR = '/deepcell_data'
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
MODEL_DIR = os.path.join(ROOT_DIR, 'models', EXP_NAME)

DATA_DIR = os.path.join(ROOT_DIR, 'users/willgraf/mesmer_retrain')
TIFF_PATH = os.path.join(DATA_DIR, 'tissue_net/seed{}'.format(SEED))

TRAIN_DATA_FILE = os.path.join(DATA_DIR, '{}_train_512x512.npz'.format(NPZ_NAME))
VAL_DATA_FILE = os.path.join(DATA_DIR, '{}_val_256x256.npz'.format(NPZ_NAME))
TEST_DATA_FILE = os.path.join(DATA_DIR, '{}_test_256x256.npz'.format(NPZ_NAME))

TEST_PRED_DATA_FILE = os.path.join(DATA_DIR, '{}_test_pred.npz'.format(NPZ_NAME))


def save_as_tiffs(npz_path, tiff_dir):
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']

    assert X.shape[0] == y.shape[0], 'X and y should have the same number of images.'

    for i in range(X.shape[0]):
        img_filename = '{:04d}_img.tif'.format(i)
        mask_filename = '{:04d}_masks.tif'.format(i)

        tifffile.imsave(os.path.join(tiff_dir, img_filename), X[i])
        tifffile.imsave(os.path.join(tiff_dir, mask_filename), y[i])
    print('saved %s files to %s' % (len(X), tiff_dir))


if __name__ == '__main__':
    data_files = [
        ('train', TRAIN_DATA_FILE),
        ('val', VAL_DATA_FILE),
        ('test', TEST_DATA_FILE),
    ]
    for prefix, data_file in data_files:
        f = os.path.join(DATA_DIR, data_file)
        subdir = os.path.join(TIFF_PATH, prefix)
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        save_as_tiffs(f, subdir)

    X_train = train_data['X']
    y_train = train_data['y']

    X_val = val_data['X']
    y_val = val_data['y']

    X_test = test_data['X']
    y_test = test_data['y']
