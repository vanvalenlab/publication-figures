import os
import copy
import math

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.io as io
import networkx as nx

from skimage.transform import resize
from skimage.segmentation import find_boundaries
from skimage.future import graph
from datetime import datetime, timedelta
from skimage.measure import regionprops_table

from skimage.exposure import rescale_intensity
from deepcell_toolbox.metrics import Metrics
from scipy.stats import pearsonr
from matplotlib import cm
from matplotlib.colors import ListedColormap

from scipy.ndimage import gaussian_filter


def plot_mod_ap(mod_ap_list, thresholds, labels):
    df = pd.DataFrame({'iou': thresholds})

    for idx, label in enumerate(labels):
        df[label] = mod_ap_list[idx]['scores']

    fig, ax = plt.subplots()
    for label in labels:
        ax.plot('iou', label, data=df, linestyle='-', marker='o')

    ax.set_xlabel('IOU Threshold')
    ax.set_ylabel('mAP')
    ax.legend()
    fig.show()


def plot_error_types(axes, error_dicts, method_labels, error_labels, colors, ylim=None):
    data_dict = pd.DataFrame(pd.Series(error_dicts[0])).transpose()

    # create single dict with all errors
    for i in range(1, len(method_labels)):
        data_dict = data_dict.append(error_dicts[i], ignore_index=True)

    data_dict['algos'] = method_labels

    for i in range(len(error_labels)):
        barchart_helper(ax=axes[i], values=data_dict[error_labels[i]], labels=method_labels,
                        title='{} Errors'.format(error_labels[i]), colors=colors, y_lim=ylim)


def barchart_helper(ax, values, labels, title, colors, y_lim=None):

    # bars are evenly spaced based on number of categories
    positions = range(len(values))
    ax.bar(positions, values, color=colors)

    # x ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    # y ticks
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # title
    ax.set_title(title)


def plot_annotator_agreement(f1_scores_list, labels):

    fig, ax = plt.subplots()
    for i in range(len(labels)):
        current_data = f1_scores_list[i]
        x = [i] * len(current_data)
        ax.plot(x, current_data, marker='o', linestyle='none', color='blue')

    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    #ax.yaxis.set_ticks_position('left')
    #ax.xaxis.set_ticks_position('bottom')


def nuclear_expansion_pixel(label_map, expansion_radius):
    expanded_map = morph.dilation(label_map, selem=morph.disk(expansion_radius))
    return expanded_map


def nuclear_expansion_watershed(label, membrane):
    new_labels = morph.watershed(membrane, markers=label, watershed_line=False)
    return new_labels





def preprocess_overlays(img_dir):
    DNA = io.imread(os.path.join(img_dir, 'DNA.tiff'))
    DNA = resize(DNA, [DNA.shape[0] * 2, DNA.shape[1] * 2], order=3)
    DNA = DNA.astype('float32')
    DNA = DNA / np.max(DNA)

    Membrane = io.imread(os.path.join(img_dir, 'Membrane.tiff'))
    Membrane = resize(Membrane, [Membrane.shape[0] * 2, Membrane.shape[1] * 2], order=3)
    Membrane = Membrane.astype('float32')
    Membrane = Membrane / np.max(Membrane)

    labels = io.imread(os.path.join(img_dir, 'labels.tiff'))
    boundaries_sub = find_boundaries(labels.astype('int'), connectivity=1, mode='subpixel')

    # subpixel producing an image with len i * 2 - 1, need to pad with a single pixel
    boundaries = np.zeros_like(Membrane)
    boundaries[:-1, :-1] = boundaries_sub

    labels = resize(labels, [labels.shape[0] * 2, labels.shape[1] * 2],
                    order=0, preserve_range=True)
    labels = labels.astype('int16')

    io.imsave(os.path.join(img_dir, 'DNA_resized.tiff'), DNA)
    io.imsave(os.path.join(img_dir, 'Membrane_resized.tiff'), Membrane)
    io.imsave(os.path.join(img_dir, 'boundaries_resized.tiff'), boundaries)
    io.imsave(os.path.join(img_dir, 'labels_resized.tiff'), labels)


def generate_crop(img_dir, row_start, col_start, length):
    DNA = io.imread(os.path.join(img_dir, 'DNA_resized.tiff'))
    DNA_cropped = DNA[row_start:(row_start + length), col_start:(col_start + length)]

    Membrane = io.imread(os.path.join(img_dir, 'Membrane_resized.tiff'))
    Membrane_cropped = Membrane[row_start:(row_start + length), col_start:(col_start + length)]

    labels = io.imread(os.path.join(img_dir, 'labels_resized.tiff'))
    labels_cropped = labels[row_start:(row_start + length), col_start:(col_start + length)]

    boundaries = io.imread(os.path.join(img_dir, 'boundaries_resized.tiff'))
    boundaries_cropped = boundaries[row_start:(row_start + length), col_start:(col_start + length)]

    io.imsave(os.path.join(img_dir, 'DNA_cropped.tiff'), DNA_cropped)
    io.imsave(os.path.join(img_dir, 'Membrane_cropped.tiff'), Membrane_cropped)
    io.imsave(os.path.join(img_dir, 'labels_cropped.tiff'), labels_cropped)
    io.imsave(os.path.join(img_dir, 'boundaries_cropped.tiff'), boundaries_cropped)


def color_labels_by_graph(labels):
    label_graph = graph.RAG(label_image=labels)
    graph_dict = nx.coloring.greedy_color(label_graph, strategy='largest_first')

    label_outline = find_boundaries(labels.astype('int'), connectivity=1)
    output_labels = copy.copy(labels)

    output_labels[label_outline > 0] = 0

    for idx in np.unique(output_labels):
        mask = output_labels == idx
        if idx == 0:
            output_labels[mask] = 0
        else:
            val = graph_dict[idx]
            output_labels[mask] = val + 1

    return output_labels


def recolor_labels(mask, values=None):
    if values is None:
        values = [100, 130, 160, 190, 220, 250]

    for idx, value in enumerate(values):
        mask[mask == (idx + 1)] = value

    return mask


def generate_inset(img_dir, row_start, col_start, length, inset_num, thickness=2):
    DNA = io.imread(os.path.join(img_dir, 'DNA_cropped.tiff'))
    DNA_inset = DNA[row_start:(row_start + length), col_start:(col_start + length)]

    Membrane = io.imread(os.path.join(img_dir, 'Membrane_cropped.tiff'))
    Membrane_inset = Membrane[row_start:(row_start + length), col_start:(col_start + length)]

    labels = io.imread(os.path.join(img_dir, 'labels_cropped.tiff'))
    labels_inset = labels[row_start:(row_start + length), col_start:(col_start + length)]

    io.imsave(os.path.join(img_dir, 'DNA_inset_{}.tiff'.format(inset_num)), DNA_inset)
    io.imsave(os.path.join(img_dir, 'Membrane_inset_{}.tiff'.format(inset_num)), Membrane_inset)
    io.imsave(os.path.join(img_dir, 'labels_inset_{}.tiff'.format(inset_num)), labels_inset)

    inset_mask = np.zeros(DNA.shape, dtype='uint8')
    inset_mask[row_start - thickness: row_start + thickness, col_start:(col_start + length)] = 128
    inset_mask[row_start + length - thickness:row_start + length + thickness,
               col_start:(col_start + length)] = 128

    inset_mask[row_start:(row_start + length), col_start - thickness: col_start + thickness] = 128
    inset_mask[row_start:(row_start + length),
               col_start + length - thickness: col_start + length + thickness] = 128

    io.imsave(os.path.join(img_dir, 'labels_inset_mask_{}.tiff'.format(inset_num)), inset_mask)


def generate_RGB_image(red=None, green=None, blue=None, percentile_cutoffs=(5, 95)):

    if red is None:
        red = np.zeros_like(green)

    combined = np.stack((red, green, blue), axis=-1)

    rgb_output = np.zeros(combined.shape, dtype='float32')

    # rescale each channel
    for idx in range(combined.shape[2]):
        if np.max(combined[:, :, idx]) == 0:
            # don't need to rescale this channel
            pass
        else:
            percentiles = np.percentile(combined[:, :, idx][combined[:, :, idx] > 0],
                                        [percentile_cutoffs[0], percentile_cutoffs[1]])
            rescaled_intensity = rescale_intensity(combined[:, :, idx].astype('float32'),
                                                   in_range=(percentiles[0], percentiles[1]),
                                                   out_range='float32')
            rgb_output[:, :, idx] = rescaled_intensity
    return rgb_output


def calculate_human_f1_scores(image_list):
    """Computes pairwise F1 scores from labeled images

    Args:
        image_list: list of predicted labels

    Returns:
        list: f1 scores for images
    """

    f1_list = []
    # loop over images to get first image
    for img1_idx in range(len(image_list) - 1):
        img1 = image_list[img1_idx]
        img1 = np.expand_dims(img1, axis=0)

        # loop over subsequent images to get corresponding predicted image
        for img2_idx in range(img1_idx + 1, len(image_list)):
            img2 = image_list[img2_idx]
            img2 = np.expand_dims(img2, axis=0)
            m = Metrics('human vs human', seg=False)
            m.calc_object_stats(y_true=img1, y_pred=img2)
            recall = m.stats['correct_detections'].sum() / m.stats['n_true'].sum()
            precision = m.stats['correct_detections'].sum() / m.stats['n_pred'].sum()
            f1 = 2 * precision * recall / (precision + recall)
            f1_list.append(f1)

    return f1_list


def calculate_alg_f1_scores(image_list, alg_pred):
    """Compare human annotations with algorithm for a given FOV

    Args:
        image_list: list of annotations from different human labelers
        alg_pred: prediction from alogrithm

    Returns:
        list: f1 scores for humans vs alg
    """

    f1_list_alg = []
    for true_img in image_list:
        true_img = np.expand_dims(true_img, axis=0)
        m = Metrics('human vs alg', seg=False)
        m.calc_object_stats(y_true=true_img, y_pred=alg_pred)
        recall = m.stats['correct_detections'].sum() / m.stats['n_true'].sum()
        precision = m.stats['correct_detections'].sum() / m.stats['n_pred'].sum()
        f1 = 2 * precision * recall / (precision + recall)
        f1_list_alg.append(f1)

    return f1_list_alg


def create_f1_score_grid(category_dicts, names):
    """Create a grid of f1 scores across different models"""

    vals = np.zeros((len(names), len(names)))

    grid = pd.DataFrame(vals, columns=names, index=names)

    for model_name in names:
        subset_dict = category_dicts[model_name]
        for dataset_name in names:
            print('current dataset is {}'.format(dataset_name))
            if dataset_name in subset_dict:
                f1 = subset_dict[dataset_name]['f1']
                print('f1 score is {}'.format(f1))
                grid.loc[model_name, dataset_name] = f1

    return grid


def plot_heatmap(vals, x_labels, y_labels, title, cmap='gist_heat', save_path=None):

    fig, ax = plt.subplots()
    im = ax.imshow(vals, cmap=cmap)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            current_val = np.round(vals[i, j], 2)
            text = ax.text(j, i, current_val,
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path)


def calculate_annotator_time(job_report):
    """Takes a job report and calculates the total amount of time spent on the job

    Args:
        job_report: pandas array containing relevant info

    Returns:
        int: total number of seconds elapsed in job
    """

    total_time = 0

    for i in range(len(job_report)):
        start_time = job_report.loc[i, '_started_at'].split(' ')[1]
        end_time = job_report.loc[i, '_created_at'].split(' ')[1]

        try:
            start_time = datetime.strptime(start_time, '%H:%M:%S').time()
        except ValueError:
            start_time = datetime.strptime(start_time, '%H:%M').time()
        start_time = timedelta(hours=start_time.hour, minutes=start_time.minute)

        try:
            end_time = datetime.strptime(end_time, '%H:%M:%S').time()
        except ValueError:
            end_time = datetime.strptime(end_time, '%H:%M').time()

        end_time = timedelta(hours=end_time.hour, minutes=end_time.minute)

        difference = end_time - start_time
        total_time += difference.seconds

    return total_time

# TODO: switch this to return list of paired IDs. Then subsequent function takes list of paired IDs, indexes in regionprops table, and appends as appropriate


def get_matching_true_ids(true_label, pred_label):

    true_ids, pred_ids = [], []

    for pred_cell in np.unique(pred_label[pred_label > 0]):
        pred_mask = pred_label == pred_cell
        overlap_ids, overlap_counts = np.unique(true_label[pred_mask], return_counts=True)

        # get ID of the true cell that overlaps with pred cell most
        true_id = overlap_ids[np.argmax(overlap_counts)]

        true_ids.append(true_id)
        pred_ids.append(pred_cell)

    return true_ids, pred_ids


def get_cell_size(label_list, label_map):
    size_list = []
    for label in label_list:
        size = np.sum(label_map == label)
        size_list.append(size)

    return size_list


def label_image_by_ratio(true_label, pred_label, threshold=2):

    true_ids, pred_ids = get_matching_true_ids(true_label, pred_label)

    true_sizes = get_cell_size(true_ids, true_label)
    pred_sizes = get_cell_size(pred_ids, pred_label)
    fill_val = -threshold + 0.02
    disp_img = np.full_like(pred_label.astype('float32'), fill_val)
    for i in range(len(pred_ids)):
        current_id = pred_ids[i]
        true_id = true_ids[i]
        if true_id == 0:
            ratio = threshold
        else:
            ratio = np.log2(pred_sizes[i] / true_sizes[i])
        mask = pred_label == current_id
        boundaries = find_boundaries(mask, mode='inner')
        mask[boundaries > 0] = 0
        if ratio > threshold:
            ratio = threshold
        if ratio < -threshold:
            ratio = -threshold
        disp_img[mask] = ratio

    disp_img[-1, -1] = -threshold
    disp_img[-1, -2] = threshold

    return disp_img


def label_image_by_value(label_mask, label_array, value_array):

    fill_val = 0
    disp_img = np.full_like(label_mask.astype('float32'), fill_val)
    boundaries = find_boundaries(label_mask, mode='inner')

    prop_df = pd.DataFrame(regionprops_table(label_mask, properties=['label', 'coords']))
    for i in range(len(label_array)):
        current_id = label_array[i]
        current_val = value_array[i]
        current_coords = prop_df.loc[prop_df['label'] == current_id, 'coords'].values
        disp_img[tuple(current_coords[0].T)] = current_val
    disp_img[boundaries > 0] = fill_val

    return disp_img



def apply_colormap_to_img(label_img):
    coolwarm = cm.get_cmap('coolwarm', 256)
    newcolors = coolwarm(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0, 1])
    newcolors[1:2, :] = black
    newcmp = ListedColormap(newcolors)

    transformed = np.copy(label_img)
    transformed -= np.min(transformed)
    transformed /= np.max(transformed)

    transformed = newcmp(transformed)

    return transformed


def get_paired_cell_ids(true_label, pred_label):

    true_ids, pred_ids = [], []

    for true_cell in np.unique(true_label[true_label > 0]):
        true_mask = true_label == true_cell
        overlap_ids, overlap_counts = np.unique(pred_label[true_mask], return_counts=True)

        # get ID of the pred cell that overlaps with true cell most
        pred_id = overlap_ids[np.argmax(overlap_counts)]

        true_ids.append(true_cell)
        pred_ids.append(pred_id)

    return true_ids, pred_ids


def get_paired_metrics(true_ids, pred_ids, true_metrics, pred_metrics):
    true_col_names = [name + '_true' for name in true_metrics.columns]
    pred_col_names = [name + '_pred' for name in pred_metrics.columns]
    col_names = true_col_names + pred_col_names

    paired_df = pd.DataFrame()

    for idx in range(len(true_ids)):
        true_cell, pred_cell = true_ids[idx], pred_ids[idx]

        try:
            true_vals = true_metrics.loc[true_metrics['label'] == true_cell].values[0]
            if pred_cell == 0:
                pred_vals = np.zeros_like(true_vals)
            else:
                pred_vals = pred_metrics.loc[pred_metrics['label'] == pred_cell].values[0]

            vals = np.append(true_vals, pred_vals)
            current_df = pd.DataFrame(data=[vals], columns=col_names)
            paired_df = paired_df.append(current_df)
        except:
            print('true_metrics label is {}'.format(true_cell))

    return paired_df


def generate_morphology_metrics(true_labels, pred_labels, properties):
    properties_df = pd.DataFrame()

    for i in range(true_labels.shape[0]):
        true_label = true_labels[i, :, :, 0]
        pred_label = pred_labels[i, :, :, 0]

        if np.max(true_label) == 0 or np.max(pred_label) == 0:
            continue

        true_props_table = regionprops_table(true_label, properties=properties)
        pred_props_table = regionprops_table(pred_label, properties=properties)

        properties_dict = {}
        for prop in properties[1:]:
            true_prop, pred_prop = get_paired_regionprops(true_label=true_label, pred_label=pred_label,
                                                  true_props_table=true_props_table,
                                                  pred_props_table=pred_props_table,
                                                  field=prop)
            properties_dict[prop + '_true'] = true_prop
            properties_dict[prop + '_pred'] = pred_prop

        properties_df = properties_df.append(pd.DataFrame(properties_dict))

    return properties_df


# create density scatter
def create_density_scatter(ax, true_vals, predicted_vals):
    from scipy.stats import gaussian_kde

    # Calculate the point density
    xy = np.vstack([true_vals, predicted_vals])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = true_vals[idx], predicted_vals[idx], z[idx]
    ax.scatter(x, y, c=z, s=50, edgecolor='')


def label_morphology_scatter(ax, true_vals, pred_vals):
    x = np.arange(0, np.max(true_vals))
    ax.plot(x, x, '-', color='red')
    p_r, _ = pearsonr(true_vals, pred_vals)
    x_pos = np.max(true_vals) * 0.05
    y_pos = np.max(pred_vals) * 0.9
    ax.text(x_pos, y_pos, 'Pearson Correlation: {}'.format(np.round(p_r, 2)))
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')


def compute_morphology_metrics(true_labels, pred_labels, properties=None):
    if properties is None:
        properties = ['label', 'area', 'major_axis_length', 'minor_axis_length']

    prop_df = pd.DataFrame()
    for idx in range(true_labels.shape[0]):
        pred_label = pred_labels[idx, :, :, 0]
        true_label = true_labels[idx, :, :, 0]

        true_ids, pred_ids = get_paired_cell_ids(true_label=true_label,
                                                         pred_label=pred_label)

        true_props_table = pd.DataFrame(regionprops_table(true_label, properties=properties))
        pred_props_table = pd.DataFrame(regionprops_table(pred_label, properties=properties))

        paired_df = get_paired_metrics(true_ids=true_ids, pred_ids=pred_ids,
                                               true_metrics=true_props_table,
                                               pred_metrics=pred_props_table)
        paired_df['img_num'] = idx
        prop_df = prop_df.append(paired_df)

    return prop_df


def get_skew_cells(input_df):
    ratio = input_df['major_axis_length_true'].values / input_df['minor_axis_length_true'].values
    skew_idx = np.logical_or(ratio < 0.6, ratio > 1.5)
    nonzero_idx = input_df['area_pred'] > 0
    combined_idx = skew_idx * nonzero_idx

    true_size = input_df['area_true'].values[combined_idx]
    pred_size = input_df['area_pred'].values[combined_idx]

    return true_size, pred_size


def get_round_cells(input_df):
    ratio = input_df['major_axis_length_true'].values / input_df['minor_axis_length_true'].values
    round_idx = np.logical_and(ratio > 0.8, ratio < 1.2)
    nonzero_idx = input_df['area_pred'] > 0
    combined_idx = round_idx * nonzero_idx

    true_size = input_df['area_true'].values[combined_idx]
    pred_size = input_df['area_pred'].values[combined_idx]

    return true_size, pred_size


def get_nonzero_cells(input_df):
    nonzero_idx = input_df['area_pred'] > 0
    true_size = input_df['area_true'].values[nonzero_idx]
    pred_size = input_df['area_pred'].values[nonzero_idx]

    return true_size, pred_size


def create_f1_score_long_df(data_array, unique_subsets):
    subset_list = []
    model_list = []
    f1_list = []

    for subset in unique_subsets:
        custom_score = data_array.loc[subset, subset]
        general_score = data_array.loc['all', subset]

        # add custom score
        subset_list.append(subset)
        model_list.append('custom')
        f1_list.append(custom_score)

        # add general score
        subset_list.append(subset)
        model_list.append('general')
        f1_list.append(general_score)

    return subset_list, model_list, f1_list

#
#
# x = np.arange(0, np.max(df[prop_name + '_true'].values))
# ax[df_idx, i].plot(x, x, '-', color='red')
# p_r, _ = pearsonr(true_vals, predicted_vals)
# x_pos = np.max(true_vals) * 0.05
# y_pos = np.max(predicted_vals) * 0.9
# ax[df_idx, i].text(x_pos, y_pos, 'Pearson Correlation: {}'.format(np.round(p_r, 2)))
# ax.set_xlabel('True Value')
# ax.set_ylabel('Predicted Value')


def calc_nuc_dist(cell_centroids, nuc_centroids):
    x1s, y1s = cell_centroids
    x2s, y2s = nuc_centroids
    dist = np.sqrt((x2s - x1s) ** 2 + (y2s - y1s) ** 2)
    return dist


def calc_jaccard_index_object(metric_predictions, true_labels, pred_labels):
    jacc_list = []
    for i in range(true_labels.shape[0]):
        y_true = true_labels[i, :, :, 0]
        y_pred = pred_labels[i, :, :, 0]
        true_ids = metric_predictions[i][0]['correct']['y_true']
        pred_ids = metric_predictions[i][0]['correct']['y_pred']

        current_accum = []

        for id in range(len(true_ids)):
            true_mask = y_true == true_ids[id]
            pred_mask = y_pred == pred_ids[id]

            current_jacc = (np.sum(np.logical_and(true_mask, pred_mask)) /
                np.sum(np.logical_or(true_mask, pred_mask)))
            current_accum.append(current_jacc)

        jacc_list.append(current_accum)
    return jacc_list


def rescale_channels(input_image, channels_last=True):
    output_image = np.zeros_like(input_image)
    for i in range(input_image.shape[-1]):
        img = input_image[..., i]
        img = img - np.min(img)
        img_max = np.max(img)
        if img_max > 0:
            img = img / img_max
        output_image[..., i] = img
    return output_image


def crop_helper(image_stack, crop_size):
    """"Helper function to take an image, and return crops of size crop_size
    Args:
        image_stack (np.array): A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int): Size of the crop to take from the image. Assumes square crops
    Returns:
        cropped_images (np.array): A 4D numpy array of shape (crops, rows, columns, channels)"""

    if len(image_stack.shape) != 4:
        raise ValueError("Incorrect dimensions of input image. "
                         "Expecting 3D, got {}".format(image_stack.shape))

    # figure out number of crops for final image
    crop_num_row = math.ceil(image_stack.shape[1] / crop_size)
    crop_num_col = math.ceil(image_stack.shape[2] / crop_size)
    cropped_images = np.zeros(
        (crop_num_row * crop_num_col * image_stack.shape[0],
            crop_size, crop_size, image_stack.shape[3]),
        dtype=image_stack.dtype)

    # Determine if image will need to be padded with zeros due to uneven division by crop
    if image_stack.shape[1] % crop_size != 0 or image_stack.shape[2] % crop_size != 0:
        # create new array that is padded by one crop size on image dimensions
        new_shape = (image_stack.shape[0], image_stack.shape[1] + crop_size,
                     image_stack.shape[2] + crop_size, image_stack.shape[3])
        new_stack = np.zeros(new_shape, dtype=image_stack.dtype)
        new_stack[:, :image_stack.shape[1], :image_stack.shape[2], :] = image_stack
        image_stack = new_stack

    # iterate through the image row by row, cropping based on identified threshold
    img_idx = 0
    for point in range(image_stack.shape[0]):
        for row in range(crop_num_row):
            for col in range(crop_num_col):
                cropped_images[img_idx, :, :, :] = \
                    image_stack[point, (row * crop_size):((row + 1) * crop_size),
                                (col * crop_size):((col + 1) * crop_size), :]
                img_idx += 1

    return cropped_images


def crop_image_stack(image_stack, crop_size, stride_fraction):
    """Function to generate a series of tiled crops across an image.
    The tiled crops can overlap each other, with the overlap between tiles determined by
    the stride fraction. A stride fraction of 0.333 will move the window over 1/3 of
    the crop_size in x and y at each step, whereas a stride fraction of 1 will move
    the window the entire crop size at each iteration.
    Args:
        image_stack (np.array): A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int): size of the crop to take from the image. Assumes square crops
        stride_fraction (float): the relative size of the stride for overlapping
            crops as a function of the crop size.
    Returns:
        cropped_images (np.array): A 4D numpy array of shape(crops, rows, cols, channels)"""

    if len(image_stack.shape) != 4:
        raise ValueError("Incorrect dimensions of input image. "
                         "Expecting 3D, got {}".format(image_stack.shape))

    if crop_size > image_stack.shape[1]:
        raise ValueError("Invalid crop size: img shape is {} "
                         "and crop size is {}".format(image_stack.shape, crop_size))

    if stride_fraction > 1:
        raise ValueError("Invalid stride fraction. Must be less than 1, "
                         "passed a value of {}".format(stride_fraction))

    # Determine how many distinct grids will be generated across the image
    stride_step = math.floor(crop_size * stride_fraction)
    num_strides = math.floor(1 / stride_fraction)

    for row_shift in range(num_strides):
        for col_shift in range(num_strides):

            if row_shift == 0 and col_shift == 0:
                # declare data holder
                cropped_images = crop_helper(image_stack, crop_size)
            else:
                # crop the image by the shift prior to generating grid of crops
                img_shift = image_stack[:, (row_shift * stride_step):,
                                        (col_shift * stride_step):, :]
                # print("shape of the input image is {}".format(img_shift.shape))
                temp_images = crop_helper(img_shift, crop_size)
                cropped_images = np.append(cropped_images, temp_images, axis=0)

    return cropped_images


def compute_cells_per_split(split_dict, keys):
    cell_count_list = []
    for key in keys:
        current_labels = split_dict[key].item()['y']
        cell_count = 0

        if key == '1':
            current_labels = current_labels[:1]
        if key == '3':
            current_labels = current_labels[:3]

        for img in range(current_labels.shape[0]):
            current_img = current_labels[img, :, :, 0]
            cells = len(np.unique(current_img)) - 1
            cell_count += cells

        cell_count_list.append(cell_count)

    return cell_count_list
