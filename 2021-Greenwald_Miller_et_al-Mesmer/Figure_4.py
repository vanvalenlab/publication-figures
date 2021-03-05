import os
import shutil

import pandas as pd
import numpy as np
import xarray as xr
import skimage.io as io
import matplotlib.pyplot as plt
import seaborn as sns

from ark.utils.load_utils import load_imgs_from_tree, load_imgs_from_dir
from ark.utils import data_utils, io_utils, load_utils
from ark.utils.io_utils import list_folders, list_files
from ark.segmentation.marker_quantification import create_marker_count_matrices, compute_marker_counts
from skimage.segmentation import find_boundaries
import figures
from ark.utils import segmentation_utils
from matplotlib.colors import ListedColormap

from skimage import morphology

from skimage.measure import regionprops_table
from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

base_dir = '/Users/noahgreenwald/Downloads/publications_data_folder/'
plot_dir = base_dir + 'plots'
data_dir = base_dir + '/phenotyping_data/pred_labels/'

# subcellular localization data
cell_labels = xr.open_dataarray(os.path.join(data_dir, 'segmentation_labels_cell.xr'))
nuc_labels = xr.open_dataarray(os.path.join(data_dir, 'segmentation_labels_nuc.xr'))
channel_data = xr.load_dataarray(os.path.join(data_dir, 'deepcell_input.xr'))

# Figure 4a
row_coords = [(200, 500), (700, 1000), (700, 1000),
              [580, 880], [724, 1024], [230, 530]]
col_coords = [(200, 500), (100, 400), (500, 800),
              [260, 560], [650, 950], [30, 330]]
points = ['Point2203', 'Point2203', 'Point2205',
          'Point4411', 'Point4119', 'Point6206']
channels = ['Ki67.tif', 'P.tif', 'ECAD.tif',
            'CD44.tif', 'PanKRT.tif', 'HER2.tif']

# create overlays
for i in range(len(channels)):
    row_start, row_end = row_coords[i]
    col_start, col_end = col_coords[i]
    current_point = points[i]
    current_channel_name = channels[i]
    current_channel = io.imread(os.path.join(data_dir, 'fovs', current_point, current_channel_name))
    current_channel = current_channel[row_start:row_end, col_start:col_end]
    current_channel = current_channel / np.max(current_channel)
    DNA = channel_data.loc[current_point, row_start:row_end, col_start:col_end, 'HH3']
    DNA = DNA / np.max(DNA)
    carbon = io.imread(os.path.join(data_dir, 'fovs', current_point, 'C.tif'))
    carbon = carbon[row_start:row_end, col_start:col_end]
    carbon = carbon / np.max(carbon)
    current_cell_label = cell_labels.loc[current_point, row_start:row_end, col_start:col_end, 'whole_cell']
    current_cell_boundary = find_boundaries(current_cell_label, mode='inner')
    current_nuc_label = nuc_labels.loc[current_point, row_start:row_end, col_start:col_end, 'nuclear']
    current_nuc_boundary = find_boundaries(current_nuc_label, mode='inner')

    combined_mask = np.zeros(current_cell_label.shape + (3,), dtype='uint8')
    combined_mask[current_cell_label > 0, 2] = 136
    combined_mask[current_cell_label > 0, 1] = 136

    nuc_mask = np.zeros_like(current_nuc_label)
    nuc_mask[current_nuc_label > 0] = 130
    nuc_mask[current_nuc_boundary] = 0
    combined_mask[current_cell_boundary, :] = 0
    combined_mask[nuc_mask > 0, 0] = 182
    combined_mask[nuc_mask > 0, 2] = 182
    combined_mask[nuc_mask > 1, 1] = 0

    save_dir = os.path.join(plot_dir, 'Figure_4a_crop_{}'.format(i))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    io.imsave(os.path.join(save_dir, 'DNA.tiff'), DNA.astype('float32'))
    io.imsave(os.path.join(save_dir, current_channel_name), current_channel)
    io.imsave(os.path.join(save_dir, 'overlay_rgb.png'), combined_mask)
    io.imsave(os.path.join(save_dir, 'Carbon.tiff'), carbon.astype('float32'))


# Figure 4b
# extract data and create outlines for visualization
cell_labels = xr.open_dataarray(os.path.join(data_dir, 'segmentation_labels_cell.xr'))
nuc_labels = xr.open_dataarray(os.path.join(data_dir, 'segmentation_labels_nuc.xr'))

# compute subcellular localization of imaging signal
img_list = ['CD44.tif', 'COX2.tif', 'ECAD.tif', 'GLUT1.tif', 'HER2.tif', 'HH3.tif',
            'Ki67.tif', 'P.tif', 'PanKRT.tif', 'pS6.tif']

folders = io_utils.list_folders(data_dir, 'Point')

fovs = io_utils.list_folders(os.path.join(data_dir, 'fovs'), 'Point')


# Since each point has different channels, we need to segment them one at a time
segmentation_labels = xr.DataArray(np.concatenate((cell_labels.values,
                                                   nuc_labels.values),
                                                   axis=-1),
                                   coords=[cell_labels.fovs,
                                           cell_labels.rows,
                                           cell_labels.cols,
                                           ['whole_cell', 'nuclear']],
                                   dims=cell_labels.dims)

compartment_df_pred_raw = pd.DataFrame()
compartment_df_pred_norm = pd.DataFrame()

# # this code require a modified version of the pipeline to run
# for fov in segmentation_labels.fovs.values:
#     channel_data = load_imgs_from_tree(os.path.join(data_dir, 'fovs'), fovs=[fov],
#                                                   img_sub_folder='potential_channels')
#
#     current_labels = segmentation_labels.loc[[fov], :, :, :]
#
#     normalized, transformed, raw = create_marker_count_matrices(
#         segmentation_labels=current_labels,
#         image_data=channel_data,
#         nuclear_counts=True,
#         split_large_nuclei=True
#     )
#     compartment_df_pred_raw = compartment_df_pred_raw.append(raw, sort=False)
#     compartment_df_pred_norm = compartment_df_pred_norm.append(normalized, sort=False)
#
# compartment_df_pred_raw.to_csv(os.path.join(data_dir, 'single_cell_data_raw.csv'), index=False)
# compartment_df_pred_norm.to_csv(os.path.join(data_dir, 'single_cell_data_norm.csv'), index=False)

# read in segmented data
compartment_df_pred_raw = pd.read_csv(os.path.join(data_dir, 'single_cell_data_raw.csv'))

channels = np.array(['CD44', 'ECAD', 'GLUT1', 'HER2', 'HH3', 'Ki67', 'P', 'PanKRT'])
nc_ratio_vals = []
nc_ratio_channels = []

# compute compartment ratio
for i in range(len(channels)):
    chan_name = channels[i]
    channel_counts = compartment_df_pred_raw.loc[:, [chan_name, chan_name + '_nuclear',
                                                     'area', 'area_nuclear']]
    cutoff = np.percentile(channel_counts.values[channel_counts.values[:, 0] > 0, 0], [20])[0]
    cutoff_idx = channel_counts[chan_name] > cutoff
    nucleated_idx = compartment_df_pred_raw['cell_size_nuclear'] > 0
    keep_idx = np.logical_and(cutoff_idx, nucleated_idx)

    channel_counts = channel_counts.loc[keep_idx, :]
    cyto_counts = channel_counts.values[:, 0] - channel_counts.values[:, 1]

    # avoid divide by zero issues
    cyto_counts[cyto_counts <= 0] = 1
    cyto_size = channel_counts.values[:, 2] - channel_counts.values[:, 3]
    cyto_size[cyto_size <= 0] = 1

    cyto_norm = cyto_counts / cyto_size

    nuc_norm = channel_counts.values[:, 1] / channel_counts.values[:, 3]

    ratio = nuc_norm / cyto_norm

    # avoid log(0) issues
    ratio[ratio == 0] = 0.01

    # cap ratio
    nc_ratio_vals.append(ratio)
    nc_ratio_channels.append([chan_name for x in range(len(cyto_counts))])

nc_ratio_vals = np.concatenate(nc_ratio_vals, axis=0)
nc_ratio_channels = np.concatenate(nc_ratio_channels, axis=0)
subcellular_df = pd.DataFrame({'channel': nc_ratio_channels, 'value': np.log2(nc_ratio_vals)})

fig, ax = plt.subplots()
ax = sns.barplot(data=subcellular_df,  x='channel', y='value', color='blue',
                order=['ECAD', 'CD44', 'HER2', 'Ki67', 'P', 'HH3'])
ax.set(ylim=(-2, 6))

# hatch color
for idx, bar in enumerate(ax.patches):
    bar.set_hatch('//')
    if idx < 3:
        bar.set_edgecolor('blue')
    else:
        bar.set_edgecolor('red')

plt.savefig(os.path.join(plot_dir, 'Figure_4b_predicted.pdf'))


# compute subcellular localization in ground truth labels
data_dir = base_dir + '/phenotyping_data/true_labels/'
true_cell_labels = xr.load_dataarray(data_dir + 'segmentation_labels_cell.xr')
true_nuc_labels = xr.load_dataarray(data_dir + 'segmentation_labels_nuc.xr')

true_segmentation_labels = xr.DataArray(np.concatenate((true_cell_labels.values.astype('int16'),
                                                   true_nuc_labels.values.astype('int16')),
                                                   axis=-1),
                                   coords=[true_cell_labels.fovs,
                                           true_cell_labels.rows,
                                           true_cell_labels.cols,
                                           ['whole_cell', 'nuclear']],
                                   dims=true_cell_labels.dims)

compartment_df_true_raw = pd.DataFrame()
compartment_df_true_norm = pd.DataFrame()

# for fov in true_segmentation_labels.fovs.values:
#     channel_data = load_imgs_from_tree(base_dir + 'cropped_inputs', fovs=[fov])
#
#     current_labels = true_segmentation_labels.loc[[fov], :, :, :]
#
#     normalized, transformed, raw = create_marker_count_matrices(
#         segmentation_labels=current_labels,
#         image_data=channel_data,
#         nuclear_counts=True,
#         split_large_nuclei=True
#     )
#     compartment_df_true_raw = compartment_df_true_raw.append(raw, sort=False)
#     compartment_df_true_norm = compartment_df_true_norm.append(normalized, sort=False)
#
# compartment_df_true_raw.to_csv(os.path.join(save_dir, 'single_cell_data_raw.csv'), index=False)
# compartment_df_true_norm.to_csv(os.path.join(save_dir, 'single_cell_data_norm.csv'), index=False)

# read in segmented data
compartment_df_true_raw = pd.read_csv(os.path.join(data_dir, 'single_cell_data_true.csv'))
# cell_counts = cell_counts.loc[cell_counts['cell_size_nuclear'] > 20, :]

channels = np.array(['CD44', 'ECAD', 'GLUT1', 'HER2', 'HH3', 'Ki67', 'P', 'PanKRT'])
nc_ratio_vals = []
nc_ratio_channels = []

# compute nuclear fraction
for i in range(len(channels)):
    chan_name = channels[i]
    channel_counts = compartment_df_true_raw.loc[:, [chan_name, chan_name + '_nuclear',
                                                     'area', 'area_nuclear']]
    cutoff = np.percentile(channel_counts.values[channel_counts.values[:, 0] > 0, 0], [20])[0]
    cutoff_idx = channel_counts[chan_name] > cutoff
    nucleated_idx = compartment_df_true_raw['cell_size_nuclear'] > 0
    keep_idx = np.logical_and(cutoff_idx, nucleated_idx)

    channel_counts = channel_counts.loc[keep_idx, :]
    cyto_counts = channel_counts.values[:, 0] - channel_counts.values[:, 1]

    # avoid divide by zero issues
    cyto_counts[cyto_counts <= 0] = 1
    cyto_size = channel_counts.values[:, 2] - channel_counts.values[:, 3]
    cyto_size[cyto_size <= 0] = 1

    cyto_norm = cyto_counts / cyto_size

    nuc_norm = channel_counts.values[:, 1] / channel_counts.values[:, 3]

    ratio = nuc_norm / cyto_norm

    # avoid log(0) issues
    ratio[ratio == 0] = 0.01

    # cap ratio
    nc_ratio_vals.append(ratio)
    nc_ratio_channels.append([chan_name for x in range(len(cyto_counts))])

nc_ratio_vals_true = np.concatenate(nc_ratio_vals, axis=0)
nc_ratio_channels_true = np.concatenate(nc_ratio_channels, axis=0)
subcellular_df_true = pd.DataFrame({'channel': nc_ratio_channels_true,
                               'value': np.log2(nc_ratio_vals_true)})

fig, ax = plt.subplots()
g = sns.barplot(data=subcellular_df_true,  x='channel', y='value', color='blue',
                order=['ECAD', 'CD44', 'HER2','Ki67', 'P', 'HH3'])

g.set(ylim=(-2, 6))
plt.savefig(os.path.join(plot_dir, 'Figure_4b_true.pdf'))

# NC ratio

# # Figure 4c
# data_dir = base_dir + 'image_files/test_split_predictions/'
# true_dict = np.load(data_dir + '20201018_multiplex_final_seed_1_nuclear_test_256x256.npz')
# X_true = true_dict['X']
# pred_cell_labels = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_cell_prediction.npz'))['y']
#
# # top crop
# io.imsave(os.path.join(plot_dir, 'Figure_4c_top_DNA.tiff'), X_true[548, 116:, :, 0].astype('float32'))
# io.imsave(os.path.join(plot_dir, 'Figure_4c_top_Membrabe.tiff'), X_true[548, 116:, :, 1].astype('float32'))
# outline_top_cell = find_boundaries(pred_cell_labels[548, 116:, :, 0], mode='inner')
# io.imsave(os.path.join(plot_dir, 'Figure_4c_top_cell_outline.tiff'), outline_top_cell.astype('float32'))
#
# # bottom crop
# io.imsave(os.path.join(plot_dir, 'Figure_4c_bottom_DNA.tiff'), X_true[853, 90:230, :, 0].astype('float32'))
# io.imsave(os.path.join(plot_dir, 'Figure_4c_bottom_Membrabe.tiff'), X_true[853, 90:230, :, 1].astype('float32'))
# outline_top_cell = find_boundaries(pred_cell_labels[853, 90:230, :, 0], mode='inner')
# io.imsave(os.path.join(plot_dir, 'Figure_4c_bottom_cell_outline.tiff'), outline_top_cell.astype('float32'))

# figure 4d
# true_dict = np.load(data_dir + '20201018_multiplex_final_seed_1_nuclear_test_256x256.npz')
# tissue_list = np.load(data_dir + '/tissue_list.npz.npy')
# true_cell_labels = true_dict['y'][..., :1].astype('int16')
# true_nuc_labels = true_dict['y'][..., 1:].astype('int16')
# pred_cell_labels = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_cell_prediction.npz'))['y']
# pred_nuc_labels = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_nuc_prediction.npz'))['y']
#
# for i in range(true_cell_labels.shape[0]):
#     img = true_cell_labels[i, :, :, 0]
#     img = morphology.label(img)
#     true_cell_labels[i, :, :, 0] = img
#
# for i in range(pred_cell_labels.shape[0]):
#     img = pred_cell_labels[i, :, :, 0]
#     img = morphology.label(img)
#     pred_cell_labels[i, :, :, 0] = img
#
# properties_df = pd.read_csv(os.path.join(data_dir, 'cell_properties.csv'))
#
# label_true_nuc, label_pred_nuc, area_true_nuc, area_pred_nuc = [], [], [], []
# for idx in range(true_nuc_labels.shape[0]):
#     true_nuc_label = true_nuc_labels[idx, :, :, 0]
#     pred_nuc_label = pred_nuc_labels[idx, :, :, 0]
#
#     true_cell_label = true_cell_labels[idx, :, :, 0]
#     pred_cell_label = pred_cell_labels[idx, :, :, 0]
#
#     true_nuc_props = pd.DataFrame(regionprops_table(true_nuc_label, properties=['label', 'area', 'centroid', 'coords']))
#     pred_nuc_props = pd.DataFrame(regionprops_table(pred_nuc_label, properties=['label', 'area', 'centroid', 'coords']))
#
#     true_cell_props = pd.DataFrame(regionprops_table(true_cell_label, properties=['label', 'coords']))
#     pred_cell_props = pd.DataFrame(regionprops_table(pred_cell_label, properties=['label', 'coords']))
#
#     current_df = properties_df.loc[properties_df['img_num'] == idx, :]
#     for true_cell in current_df['label_true'].values:
#         true_cell_coords = true_cell_props.loc[true_cell_props['label'] == true_cell, 'coords'].values[0]
#         true_nuc_id = segmentation_utils.find_nuclear_mask_id(true_nuc_label, cell_coords=true_cell_coords)
#         if true_nuc_id is None:
#             true_nuc_area = 0
#         else:
#             true_nuc_area = true_nuc_props.loc[true_nuc_props['label'] == true_nuc_id, 'area'].values[0]
#         label_true_nuc.append(true_nuc_id)
#         area_true_nuc.append(true_nuc_area)
#
#         pred_cell = current_df.loc[current_df['label_true'] == true_cell, 'label_pred'].values[0]
#         pred_cell_coords = pred_cell_props.loc[pred_cell_props['label'] == pred_cell, 'coords'].values[0]
#         pred_nuc_id = segmentation_utils.find_nuclear_mask_id(pred_nuc_label, cell_coords=pred_cell_coords)
#         if pred_nuc_id is None:
#             pred_nuc_area = 0
#         else:
#             pred_nuc_area = pred_nuc_props.loc[pred_nuc_props['label'] == pred_nuc_id, 'area'].values[0]
#         label_pred_nuc.append(pred_nuc_id)
#         area_pred_nuc.append(pred_nuc_area)
#
# properties_df['label_true_nuc'] = label_true_nuc
# properties_df['label_pred_nuc'] = label_pred_nuc
# properties_df['area_true_nuc'] = area_true_nuc
# properties_df['area_pred_nuc'] = area_pred_nuc
# properties_df['nc_ratio_true'] = properties_df['area_true_nuc'] / properties_df['area_true']
# properties_df['nc_ratio_pred'] = properties_df['area_pred_nuc'] / properties_df['area_pred']
#
# properties_df.to_csv(os.path.join(data_dir, 'cell_properties_with_new_props_nc.csv'), index=False)
#
# properties_df = pd.read_csv(os.path.join(data_dir, 'cell_properties_with_new_props_nc.csv'))
#
#
# plotting_props = ['major_minor_axis_ratio', 'perim_square_over_area', 'nc_ratio']
#
# properties_df.loc[properties_df['nc_ratio_pred'] > 1, 'nc_ratio_pred'] = 1
# properties_df.loc[properties_df['nc_ratio_true'] > 1, 'nc_ratio_true'] = 1
#
#
# mins = [0, 10, -0.25]
# maxes = [3, 25, 1.25]
# for i in range(len(plotting_props)):
#     prop = plotting_props[i]
#     x = properties_df[prop + '_true'].values
#     y = properties_df[prop + '_pred'].values
#
#     r, _ = pearsonr(x, y)
#
#     current_min, current_max = mins[i], maxes[i]
#     # Create meshgrid
#     xx, yy = np.mgrid[current_min:current_max:100j,
#              current_min:current_max:100j]
#
#     positions = np.vstack([xx.ravel(), yy.ravel()])
#     values = np.vstack([x, y])
#     kernel = st.gaussian_kde(values)
#     f = np.reshape(kernel(positions).T, xx.shape)
#
#
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.gca()
#     ax.set_xlim(current_min, current_max)
#     ax.set_ylim(current_min, current_max)
#     cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
#     l = ax.imshow(np.rot90(f), cmap='coolwarm')
#     ax.set_xlabel('True value')
#     ax.set_ylabel('Predicted Value')
#     fig.colorbar(l)
#     fig.text(xmax - 4 * deltaX, ymin + 4 * deltaY, str(np.round(r, 2)))
#
#     plt.savefig(os.path.join(plot_dir, prop + '_correlation_4d.pdf'))
#
# # create colorbar
# x = np.random.rand(400).reshape(40, 10)
# fig, ax = plt.subplots()
# x1 = ax.imshow(x, cmap='coolwarm')
# fig.colorbar(x1)
# plt.savefig(os.path.join(plot_dir, 'Figure_4d_colorbar.pdf'))


# Out of plane nuclei

# # Figure 4e
# # segmentations colored by nucleated vs not
# y_pred = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_512x512_predictions.npz'))['y']
# true_dict = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_512x512.npz'))
# X_true = true_dict['X']
#
# idx = 168
# img = X_true[idx, :, :, :]
# labels = y_pred[idx, :, :, 0]
#
# io.imsave(os.path.join(plot_dir, 'Figure_4e_DNA.tif'), img[..., 0].astype('float32'))
# io.imsave(os.path.join(plot_dir, 'Figure_4e_Membrane.tif'), img[..., 1].astype('float32'))
# io.imsave(os.path.join(plot_dir, 'Figure_4e_labels.tif'), labels)
#
# current_DNA = io.imread(os.path.join(plot_dir, 'Figure_4e_DNA.tif'))
# current_Membrane = io.imread(os.path.join(plot_dir, 'Figure_4e_Membrane.tif'))
# current_label = io.imread(os.path.join(plot_dir, 'Figure_4e_labels.tif'))
# current_boundary = find_boundaries(current_label, mode='inner')
# DNA_xr = xr.DataArray(np.expand_dims(current_DNA, axis=-1),
#                       coords=[range(512), range(512), ['DNA']],
#                       dims=['rows', 'cols', 'channels'])
#
# label_xr = xr.DataArray(np.expand_dims(current_label, axis=-1),
#                       coords=[range(512), range(512), ['whole_cell']],
#                       dims=['rows', 'cols', 'compartments'])
#
# marker_counts = compute_marker_counts(input_images=DNA_xr, segmentation_masks=label_xr)
# sns.distplot(marker_counts.loc['whole_cell', :, 'DNA'])
#
# overlay_img = np.full_like(current_label, 255, dtype='uint8')
# overlay_img[current_label > 0] = 0
#
# anucleated_cells = marker_counts.loc['whole_cell', marker_counts.loc['whole_cell', :, 'DNA'].values < 10, 'label']
#
# anucleated_mask = np.isin(current_label, anucleated_cells.values)
#
# overlay_img[anucleated_mask] = 130
# overlay_img[current_boundary] = 255
#
#
# io.imsave(os.path.join(plot_dir, 'Figure_4e_DNA_cropped.tif'), current_DNA[50:450, 100:300])
# io.imsave(os.path.join(plot_dir, 'Figure_4e_Membrane_cropped.tif'), current_Membrane[50:450, 100:300])
# io.imsave(os.path.join(plot_dir, 'Figure_4e_label_overlay_cropped.tif'), overlay_img[50:450, 100:300])
#
#
# # split RGB images after creating them in photoshop
# bottom_rgb = io.imread(os.path.join(plot_dir, 'Figure_4e_bottom_rgb.tif'))
# io.imsave(os.path.join(plot_dir, 'Figure_4e_bottom_rgb_cropped.tif'), bottom_rgb[116:, :, :])
#
# top_rgb = io.imread(os.path.join(plot_dir, 'Figure_4e_top_rgb.tif'))
# io.imsave(os.path.join(plot_dir, 'Figure_4e_top_rgb_cropped.tif'), top_rgb[90:230, :, :])
#
# # Figure 4f
# anuclear_fraction_pred = []
# for idx in np.unique(properties_df['img_num']):
#     current_counts = properties_df.loc[properties_df['img_num'] == idx]
#     anucleated_count = np.sum(current_counts['area_pred_nuc'] == 0)
#     total_count = len(current_counts)
#     anuclear_fraction_pred.append(anucleated_count/total_count)
#
# anuclear_fraction_true = []
# for idx in np.unique(properties_df['img_num']):
#     current_counts = properties_df.loc[properties_df['img_num'] == idx]
#     anucleated_count = np.sum(current_counts['area_true_nuc'] == 0)
#     total_count = len(current_counts)
#     anuclear_fraction_true.append(anucleated_count/total_count)
#
# anuclear_df = pd.DataFrame({'anuclear_frac_pred': anuclear_fraction_pred,
#                             'anuclear_frac_true': anuclear_fraction_true,
#                             'tissue': tissue_list})
#
# sums = anuclear_df.groupby('tissue')['anuclear_frac_true'].mean()
#
# # sort by decreasing nuclear counts
# anuclear_counts = sums.values
# anuclear_tissue = sums.index.values
# sort_idx = np.argsort(anuclear_counts)
# anuclear_counts, anuclear_tissue = anuclear_counts[sort_idx], anuclear_tissue[sort_idx]
#
# # remove breast and skin
# keep_idx = np.isin(anuclear_tissue, ['breast', 'immune', 'pancreas', 'gi'])
# anuclear_counts, anuclear_tissue = anuclear_counts[keep_idx], anuclear_tissue[keep_idx]
#
# fig, ax = plt.subplots()
# width = 0.35
# ax.bar(anuclear_tissue, anuclear_counts, label='Nuclear Fraction')
# # Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_title('Fraction anuclear cells')
# ax.set_ylim(0, 0.20)
# fig.savefig(plot_dir + 'Figure_4f_true.pdf')


# Cell subtype evaluation
data_dir = base_dir + 'lineage_enumeration/'
true_labels = xr.load_dataarray(os.path.join(data_dir, 'true_labels.xr'))

# segment data
mesmer_labels = load_imgs_from_dir(data_dir=data_dir + '/summed_mesmer_output',
                                   xr_channel_names=['whole_cell'], force_ints=True)

channel_data = xr.load_dataarray(os.path.join(data_dir, 'channel_data.xr'))

norm_data_pred, _ = create_marker_count_matrices(
    segmentation_labels=mesmer_labels,
    image_data=channel_data)

norm_data_true, _ = create_marker_count_matrices(
    segmentation_labels=true_labels,
    image_data=channel_data)

# set thresholds for each cell type
sns.distplot(norm_data_pred['CD68'])

marker_thresholds = {'CK': 3, 'CD8': 3, 'CD11c': 3, 'CD68': 6}

cluster_idx_pred = np.zeros(len(norm_data_pred))

for idx, marker in enumerate(['CD8', 'CK', 'CD11c']):
    threshold = marker_thresholds[marker]
    positive_idx = norm_data_pred[marker] > threshold

    # don't reassign cells that already have a value from previous round
    update_idx = np.logical_and(positive_idx, cluster_idx_pred == 0)
    cluster_idx_pred[update_idx] = idx + 1

norm_data_pred['cluster_idx'] = cluster_idx_pred + 1

cluster_idx_true = np.zeros(len(norm_data_true))

for idx, marker in enumerate(['CD8', 'CK', 'CD11c']):
    threshold = marker_thresholds[marker]
    positive_idx = norm_data_true[marker] > threshold

    # don't reassign cells that already have a value from previous round
    update_idx = np.logical_and(positive_idx, cluster_idx_true == 0)
    cluster_idx_true[update_idx] = idx + 1

norm_data_true['cluster_idx'] = cluster_idx_true + 1

# compute fraction of each cell type in each image
counts_list, cells_list, algorithm_list = [], [], []
for fov in fovs:
    current_data = norm_data_pred.loc[norm_data_pred['fov'] == fov, :]
    _, current_counts = np.unique(current_data['cluster_idx'], return_counts=True)
    current_counts =  current_counts / len(current_data)
    counts_list.append(current_counts)
    cells_list.append(['ungated', 'T cell', 'Tumor', 'Myeloid'])
    algorithm_list.append(['pred', 'pred', 'pred', 'pred'])

for fov in fovs:
    current_data = norm_data_true.loc[norm_data_true['fov'] == fov, :]
    _, current_counts = np.unique(current_data['cluster_idx'], return_counts=True)
    current_counts =  current_counts / len(current_data)
    counts_list.append(current_counts)
    cells_list.append(['ungated', 'T cell', 'Tumor', 'Myeloid'])
    algorithm_list.append(['true', 'true', 'true', 'true'])

# combine together
counts_list = np.concatenate(counts_list)
cells_list = np.concatenate(cells_list)
algorithm_list = np.concatenate(algorithm_list)

plotting_df = pd.DataFrame({'frequency': counts_list,
                            'cell_type': cells_list,
                            'type': algorithm_list})

cell_types = ['ungated', 'T cell', 'Tumor', 'Myeloid']
fig, ax = plt.subplots()

for i in range(len(cell_types)):
    cell_type = cell_types[i]
    cell_data = plotting_df.loc[plotting_df['cell_type'] == cell_type, :]
    true_counts = cell_data.loc[cell_data['type'] == 'true', :]['frequency'].values
    pred_counts = cell_data.loc[cell_data['type'] == 'pred', :]['frequency'].values

    ax.scatter(np.full(len(true_counts), i * 2), true_counts)
    ax.scatter(np.full(len(pred_counts), i * 2 + 1), pred_counts)

    for j in range(len(true_counts)):
        ax.plot([i * 2, i * 2 + 1], [true_counts[j], pred_counts[j]], c='k')


fig.savefig(os.path.join(plot_dir, 'Figure_4j.pdf'))

# create overlays with cell subtypes
new_colormap_vals = [[0, 0, 0],
                    [0.8, 0.8, 0.8],
                    [255 / 256, 77 / 256, 255 / 256],
                    [22 / 255, 22 / 256, 252 / 256],
                    [111 / 256, 190 / 256, 68 / 256]]

new_colormap = ListedColormap(new_colormap_vals)

example_fov = 'BC166_B4x10_58706_12793_component_data'
current_label = true_labels.loc['BC166_B4x10_58706_12793_component_data', :, :, 'whole_cell']
current_cluster = norm_data_true.loc[norm_data_true['fov'] == 'BC166_B4x10_58706_12793_component_data', :]
img = figures.label_image_by_value(current_label, current_cluster['label'].values, current_cluster['cluster_idx'].values)

img_cm = new_colormap(img / 4)

io.imshow(img[:500, 680:1180], cmap=new_colormap)
io.imsave(os.path.join(plot_dir, 'Figure_4i_overlay.png'), img_cm[:500, 680:1180])


example_fov = 'BC166_B4x10_58706_12793_component_data'
current_label = mesmer_labels.loc['BC166_B4x10_58706_12793_component_data', :, :, 'whole_cell']
current_cluster = norm_data_pred.loc[norm_data_pred['fov'] == 'BC166_B4x10_58706_12793_component_data', :]
img = figures.label_image_by_value(current_label, current_cluster['label'].values, current_cluster['cluster_idx'].values)

img_cm = new_colormap(img / 4)

io.imshow(img[:500, 680:1180], cmap=new_colormap)
io.imsave(os.path.join(plot_dir, 'Figure_4h_overlay.png'), img_cm[:500, 680:1180])

for channel in ['CD8', 'CK', 'CD11c', 'DAPI']:
    current_img = io.imread(os.path.join(data_dir, example_fov, channel + '.tif'))
    current_img = current_img / np.max(current_img)
    io.imsave(os.path.join(plot_dir, 'Figure_4_' + channel + '_cropped.tif'),
              current_img[:500, 680:1180])
