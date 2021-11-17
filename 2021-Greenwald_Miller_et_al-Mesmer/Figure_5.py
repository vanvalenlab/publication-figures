from matplotlib import cm

import os
import xarray as xr

import matplotlib.pyplot as plt

import skimage.io as io

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

import skimage
from skimage.segmentation import find_boundaries
from skimage.measure import label
from sklearn.preprocessing import StandardScaler
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops_table, regionprops
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap


import figures
from ark.segmentation import marker_quantification

base_dir = 'path_to_zip_folder/publications_data_folder/'
plot_dir = base_dir + 'plots'
data_dir = base_dir + '/decidua/'


segmentation_labels_selected = xr.load_dataarray(os.path.join(data_dir, 'segmentation_labels_selected.xr'))
nuc_labels_selected = xr.load_dataarray(os.path.join(data_dir, 'nuc_labels_selected.xr'))

# points for analysis
selected_fovs = ['6_31725_8_2', '6_31725_15_1', '6_31726_8_7',  '6_31727_15_3', '6_31728_15_6',
                '16_31762_20_9',  '16_31765_19_1', '16_31767_3_3', '18_31782_5_7',  '18_31785_5_13']

# create nuclear and whole-cell segmentation arrays
blank_channel_data = xr.DataArray(np.zeros((10, 2048, 2048, 1), dtype='uint8'),
                                            coords=[segmentation_labels_selected.fovs,
                                                    segmentation_labels_selected.rows,
                                                    segmentation_labels_selected.cols,
                                                    ['blank_channel']],
                                            dims=['fovs', 'rows', 'cols', 'channels'])

segmentation_labels_combined = xr.DataArray(np.concatenate((segmentation_labels_selected.values,
                                                      nuc_labels_selected.values),
                                                     axis=-1),
                                            coords=[segmentation_labels_selected.fovs,
                                                    segmentation_labels_selected.rows,
                                                    segmentation_labels_selected.cols,
                                                    ['whole_cell', 'nuclear']],
                                            dims=['fovs', 'rows', 'cols', 'compartments'])

# segment data
normalized_counts, _ = marker_quantification.create_marker_count_matrices(segmentation_labels=segmentation_labels_combined,
                                                                                       image_data=blank_channel_data,
                                                                                                   nuclear_counts=True)

normalized_counts.to_csv(os.path.join(data_dir, 'normalized_counts_with_nuc_selected.csv'), index=False)
normalized_counts = pd.read_csv(os.path.join(data_dir, 'normalized_counts_with_nuc_selected.csv'))


# generate morphology info
morph_df = pd.DataFrame()
for j in range(segmentation_labels_selected.shape[0]):
    print("current fov is {}".format(j))
    current_label = segmentation_labels_selected[j, :, :, 0].values
    current_fov = str(segmentation_labels_selected.fovs[j].values)
    props = ['label', 'area', 'eccentricity', 'major_axis_length', 'minor_axis_length',
             'perimeter', 'centroid', 'convex_area',
             'equivalent_diameter']
    current_props = pd.DataFrame(regionprops_table(current_label, properties=props))
    current_props['point'] = current_fov
    morph_df = morph_df.append(current_props)

# read in cell cluster info
cluster_df1 = pd.read_csv(os.path.join(data_dir, 'cell_table_for_NG.csv'))
cluster_df_selected = cluster_df1.loc[np.isin(cluster_df1['point'], selected_fovs), :]
cluster_id = [cluster_df_selected['point'].values[row] + '_' + str(cluster_df_selected['label'].values[row]) for row in range(len(cluster_df_selected))]
cluster_df_selected['unique_id'] = cluster_id


# generate unique ID for each cell to enable matching between DFs
morph_id = [morph_df['point'].values[row] + '_' + str(morph_df['label'].values[row]) for row in range(len(morph_df))]
morph_df['unique_id'] = morph_id
cluster_df_selected = cluster_df_selected.merge(morph_df)

# generate unique id for cell in nuclear counts df to enable merging
norm_id = [normalized_counts['fov'].values[row] + '_' + str(int(normalized_counts['label'].values[row])) for row in range(len(normalized_counts))]
normalized_counts['unique_id'] = norm_id
normalized_counts_abrev = normalized_counts.loc[:, ['unique_id', 'area_nuclear', 'label_nuclear']]

cluster_df_selected = cluster_df_selected.merge(normalized_counts_abrev)

# add composite columns
week = [cluster_df_selected['point'].values[row].split('_')[0] for row in range(len(cluster_df_selected))]
cluster_df_selected['week'] = pd.to_numeric(week)
cluster_df_selected['stage'] = ['early' if x == 6 else 'late' for x in cluster_df_selected['week']]
patient = [cluster_df_selected['point'].values[row].split('_')[1] for row in range(len(cluster_df_selected))]
cluster_df_selected['patient'] = patient


nucleated = cluster_df_selected['area_nuclear'].values > 0
cluster_df_selected['nucleated'] = nucleated

cluster_df_selected['major_minor_axis_ratio'] = cluster_df_selected['major_axis_length'] / cluster_df_selected['minor_axis_length']
cluster_df_selected['perim_square_over_area'] = np.square(cluster_df_selected['perimeter']) / cluster_df_selected['area']
cluster_df_selected['major_axis_equiv_diam_ratio'] = cluster_df_selected['major_axis_length'] / cluster_df_selected['equivalent_diameter']
cluster_df_selected['convex_hull_resid'] = (cluster_df_selected['convex_area'] - cluster_df_selected['area']) / cluster_df_selected['convex_area']
cluster_df_selected['nc_ratio'] = cluster_df_selected['area_nuclear'] / cluster_df_selected['area']
cluster_df_selected['stage'] = ['early' if np.isin(x, [6, 8]) else 'late' for x in cluster_df_selected['week']]
cluster_df_selected['nucleated'] = cluster_df_selected['area_nuclear'] > 0

# metrics that aren't computed by reginoprops
concavity_count = []
centroid_dif = []
updated_id = []


for fov in np.unique(cluster_df_selected['point']):
    current_label = segmentation_labels_combined.loc[fov, :, :, 'whole_cell'].values

    current_props = regionprops(current_label)

    for prop in current_props:
        # cell centroid shift
        cell_image = prop.image
        convex_image = prop.convex_image
        cell_M = skimage.measure.moments(cell_image)
        cell_centroid = cell_M[1, 0] / cell_M[0, 0], cell_M[0, 1] / cell_M[0, 0]

        convex_M = skimage.measure.moments(convex_image)
        convex_centroid = convex_M[1, 0] / convex_M[0, 0], convex_M[0, 1] / convex_M[0, 0]

        centroid_dist = np.sqrt((cell_centroid[1] - convex_centroid[1]) ** 2 +
                                (cell_centroid[0] - convex_centroid[0]) ** 2)

        centroid_dist /= np.sqrt(prop.area)

        # number of concavities
        diff_img = convex_image ^ cell_image
        if np.sum(diff_img) > 0:
            labeled_diff_img = label(diff_img)
            hull_prop_df = pd.DataFrame(regionprops_table(labeled_diff_img, properties=['area', 'perimeter']))
            hull_prop_df['compactness'] = np.square(hull_prop_df['perimeter']) / hull_prop_df['area']
            small_idx = np.logical_and(hull_prop_df['area'] > 10,
                                       hull_prop_df['compactness'] < 60)
            large_idx = hull_prop_df['area'] > 150
            combined_idx = np.logical_or(small_idx, large_idx)

            concavities = np.sum(combined_idx)
        else:
            concavities = 0

        concavity_count.append(concavities)
        centroid_dif.append(centroid_dist)
        updated_id.append(fov + '_' + str(prop.label))

new_df = pd.DataFrame({'unique_id': updated_id,
                       'concavity_count': concavity_count,
                       'centroid_dif': centroid_dif})


cluster_df_selected = cluster_df_selected.merge(new_df)


cluster_df_selected.to_csv(os.path.join(data_dir, 'annotated_cell_table_with_nuc_all.csv'), index=False)
cluster_df_selected = pd.read_csv(os.path.join(data_dir, 'annotated_cell_table_with_nuc_selected.csv'))




umap_metrics = ['convex_hull_resid', 'centroid_dif', 'major_axis_equiv_diam_ratio',
                     'perim_square_over_area', 'area',
                     'concavity_count']

# Figure 5a
plot_dir = base_dir + '/plots'
crop_dir = os.path.join(plot_dir, 'Figure_5a')
fov_folder = os.path.join(data_dir, '16_31762_20_9_denoised')
if not os.path.exists(crop_dir):
    os.makedirs(crop_dir)
for channel in ['H3', 'VIM', 'CD3', 'CD56', 'CD14', 'HLAG']:
    full_path = os.path.join(fov_folder, channel + '.tif')
    save_path = os.path.join(crop_dir, channel + '.tif')
    img = io.imread(full_path)
    img = img[700:1200, 700:1700]
    io.imsave(save_path, img)


# Figure 5b
fov_folder = os.path.join(data_dir, '18_31785_5_13')

row_coords = [[400, 800], [1600, 2000]]
col_coords = [[1500, 1900], [1600, 2000]]

for i in range(len(row_coords)):
    crop_dir = os.path.join(data_dir, 'selected_fovs_plots/Figure_4b_crop_{}'.format(i))
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    for channel in ['H3', 'VIM', 'CD3', 'CD56', 'CD14', 'HLAG', 'segmentation_label', 'segmentation_borders']:
        full_path = os.path.join(fov_folder, channel + '.tiff')
        save_path = os.path.join(crop_dir, channel + '.tiff')
        rows, cols = row_coords[i], col_coords[i]
        img = io.imread(full_path)
        img = img[rows[0]:rows[1], cols[0]:cols[1]]
        io.imsave(save_path, img)

        if channel == 'segmentation_label':
            disp_img = figures.label_image_by_ratio(img, img)
            disp_img_final = figures.apply_colormap_to_img(disp_img)
            io.imsave(os.path.join(crop_dir, 'label_outline.png'),
                      disp_img_final.astype('float32'))

# Figure 5d
cluster_df_selected_nucleated = cluster_df_selected.loc[cluster_df_selected['nucleated'] == True, :]

combined_fov_list = ['6_31727_15_3', '6_31727_15_3', '6_31727_15_3',
                     '6_31727_15_3', '6_31727_15_3', '6_31727_15_3',
                     '6_31727_15_3', '6_31727_15_3', '6_31727_15_3',
                     '6_31727_15_3', '6_31727_15_3', '6_31727_15_3',
                     '16_31762_20_9', '6_31727_15_3',
                     '6_31727_15_3', '6_31727_15_3', '6_31727_15_3', '6_31727_15_3',
                     '6_31727_15_3', '6_31727_15_3', '6_31727_15_3', '6_31727_15_3',
                     '6_31727_15_3', '6_31727_15_3', '6_31727_15_3'
                     ]
combined_cell_id_list = [141, 39, 245,
                         50, 643, 1110,
                         942, 1915, 1410,
                         2425, 2040, 1717,
                         927, 1215,
                         71, 89, 141, 393,
                         747, 978, 1078, 475,
                         1916, 1987, 1682
                         ]
combined_category_list = ['convex_0', 'convex_1', 'convex_2',
                          'centroid_0', 'centroid_1', 'centroid_2',
                          'concavity_0', 'concavity_1', 'concavity_2',
                          'major_axis_0', 'major_axis_1', 'major_axis_2',
                          'perimeter_0', 'perimeter_1',
                          'random_0', 'random_1', 'random_2', 'random_3', 'random_4',
                          'random_5', 'random_6', 'random_7', 'random_8', 'random_9',
                          'random_10']


# save segmentations with nuclear segmentation
for i in range(len(combined_category_list)):
    crop_dir = os.path.join(data_dir, 'Figure_4g_{}'.format(combined_category_list[i]))
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    current_fov = combined_fov_list[i]
    current_cell_id = combined_cell_id_list[i]
    idx = np.where(np.logical_and(cluster_df_selected_nucleated['point'] == current_fov,
                                  cluster_df_selected_nucleated['label'] == current_cell_id))[0][0]

    centroid_row, centroid_col, nuc_id = cluster_df_selected_nucleated.iloc[idx][['centroid-0', 'centroid-1', 'label_nuclear']]

    centroid_row = int(centroid_row)
    centroid_col = int(centroid_col)

    current_label = segmentation_labels_selected.loc[current_fov, :, :, 'whole_cell'].values
    offset = 40
    current_label_crop = current_label[centroid_row - offset: centroid_row + offset,
                                       centroid_col - offset: centroid_col + offset]
    mask = current_label_crop == current_cell_id
    segmentation_img = np.zeros((80, 80), dtype='uint8')
    segmentation_img[mask] = 255

    current_nuc_label = nuc_labels_selected.loc[current_fov, :, :, 'nuclear'].values
    offset = 40
    current_nuc_label_crop = current_nuc_label[centroid_row - offset: centroid_row + offset,
                         centroid_col - offset: centroid_col + offset]
    nuc_mask = current_nuc_label_crop == nuc_id
    nuc_segmentation_img = np.zeros((80, 80), dtype='uint8')
    nuc_segmentation_img[nuc_mask] = 255

    overlay_img = np.zeros((80, 80), dtype='uint8')
    overlay_img[mask] = 255
    overlay_img[nuc_mask] = 122

    io.imsave(os.path.join(crop_dir, 'segmentation_img_nuc.png'), nuc_segmentation_img)
    io.imsave(os.path.join(crop_dir, 'segmentation_img.png'), segmentation_img)
    io.imsave(os.path.join(crop_dir, 'segmentation_img_overlay.png'), overlay_img)

    for channel in ['H3', 'VIM', 'CD3', 'CD56', 'CD14', 'HLAG']:
        full_path = os.path.join(data_dir, current_fov, channel + '.tiff')
        save_path = os.path.join(crop_dir, channel + '.tiff')
        figures.save_image_crops(full_path, centroid_row, centroid_col, 40, save_path)

# Figure 5e
metric_list = ['convex_hull_resid', 'concavity_count', 'centroid_dif', 'convex_hull_resid','concavity_count',
               'perim_square_over_area', 'centroid_dif', 'perim_square_over_area', 'centroid_dif']
overlay_img_list = ['6_31726_8_7', '6_31725_8_2', '6_31725_8_2', '6_31725_8_2', '6_31725_15_1',
                    '6_31725_15_1', '6_31726_8_7', '18_31782_5_7', '18_31785_5_13']
overlay_img_row_coords = [[0, 500], [0, 1000], [1400, 1800], [0, 500], [1000, 1500],
                        [0, 500], [800, 1300], [0, 500], [500, 1000]]
overlay_img_col_coords = [[0, 500], [0, 1000], [0, 400], [1100, 1600], [0, 500],
                        [100, 600], [200, 700], [0, 500], [700, 1200]]

plasma_cmap = cm.get_cmap('plasma', 256)
colors = plasma_cmap(np.linspace(0, 1, 256))
black = np.array([0, 0, 0, 1])
colors[0:1, :] = black
metric_cmap = ListedColormap(colors)


for i in range(len(metric_list)):
    save_path = os.path.join(plot_dir, 'Figure_5e_{}_{}.png'.format(i, metric_list[i]))

    current_fov = overlay_img_list[i]
    current_label = segmentation_labels_selected.loc[current_fov, :, :, 'whole_cell'].values
    current_cluster = cluster_df_selected.loc[cluster_df_selected['point'] == current_fov, :]

    values = current_cluster[metric_list[i]].values
    values = values / np.max(values)
    values += 0.01
    img = figures.label_image_by_value(current_label, current_cluster['label'].values, values)
    row_coords, col_coords = overlay_img_row_coords[i], overlay_img_col_coords[i]
    img = metric_cmap(img[row_coords[0]:row_coords[1], col_coords[0]:col_coords[1]])
    io.imsave(save_path, img)


# scalebar for metrics
example_fov = '16_31762_20_9'
current_label = segmentation_labels_selected.loc[example_fov, :, :, 'whole_cell'].values
current_cluster = cluster_df_selected.loc[cluster_df_selected['point'] == example_fov, :]
img = figures.label_image_by_value(current_label, current_cluster['label'].values, current_cluster['convex_hull_resid'].values)
fig, ax = plt.subplots()
pos = ax.imshow(img, cmap=metric_cmap)
fig.colorbar(pos)
plt.savefig(os.path.join(plot_dir, 'Figure_4e_colorbar.pdf'))



## Kmeans clustering
kmeans_data = cluster_df_selected[umap_metrics]
kmeans_data_scaled = StandardScaler().fit_transform(kmeans_data)
cluster_fit = KMeans(n_clusters=4).fit(kmeans_data_scaled)
cluster_df_selected['kmeans_labels'] = cluster_fit.labels_

cluster_labels = np.zeros(len(cluster_df_selected))
cluster_labels[cluster_df_selected['kmeans_labels'] == 2] = 1
cluster_labels[cluster_df_selected['kmeans_labels'] == 3] = 2
cluster_labels[cluster_df_selected['kmeans_labels'] == 1] = 3
cluster_labels[cluster_df_selected['kmeans_labels'] == 0] = 4
cluster_df_selected['kmeans_labels'] = cluster_labels


# Figure 5f
heatmap_vals = pd.DataFrame(np.zeros((4, len(umap_metrics))), columns=umap_metrics, index=range(1, 5))
for i in range(1, 5):
    # kmeans_subset = kmeans_data_scaled[cluster_labels == i]
    kmeans_subset = kmeans_data_scaled[cluster_df_selected['kmeans_labels'] == i]

    heatmap_vals.loc[i, :] = np.nanmean(kmeans_subset, axis=0)

heatmap_vals = heatmap_vals - np.min(heatmap_vals)
heatmap_vals = heatmap_vals / np.max(heatmap_vals)
sns.clustermap(heatmap_vals, cmap='Greys')
plt.savefig(os.path.join(plot_dir, 'Figure_5f.pdf'))

# Figure 5g
# list of selected image crops for each cluster
cluster_info = {'cluster_1':
                    {'fov_list': ['18_31782_5_7', '18_31782_5_7', '18_31782_5_7', '18_31782_5_7',
                                  '18_31782_5_7',
                                  '18_31782_5_7', '18_31782_5_7', '18_31782_5_7', '18_31782_5_7',
                                  '18_31782_5_7',
                                  '18_31782_5_7', '18_31782_5_7'],
                     'cell_id_list': [1120, 1177, 1229, 1254, 1334,
                                      1385, 1457, 1904, 2008, 2054,
                                      2183, 2366]},
                'cluster_2':
                    {'fov_list': ['16_31762_20_9', '16_31762_20_9', '16_31762_20_9',
                                  '16_31762_20_9', '16_31762_20_9',
                                  '16_31762_20_9', '16_31762_20_9', '16_31762_20_9',
                                  '16_31762_20_9', '16_31762_20_9',
                                  '16_31762_20_9', '16_31762_20_9'],
                     'cell_id_list': [2095, 2119, 2239, 2246, 2249,
                                      2255, 2290, 2317, 2436, 2438,
                                      2459, 2484]},
                'cluster_3':
                    {'fov_list': ['6_31727_15_3', '6_31727_15_3', '6_31727_15_3', '6_31727_15_3',
                                  '6_31727_15_3',
                                  '6_31727_15_3', '6_31727_15_3', '6_31727_15_3', '6_31727_15_3',
                                  '6_31727_15_3',
                                  '6_31727_15_3', '6_31727_15_3'],
                     'cell_id_list': [1911, 1923, 1941, 1953, 1956,
                                      1969, 1980, 2010, 2027, 2028,
                                      2035, 2040]},
                'cluster_4':
                    {'fov_list': ['16_31762_20_9', '16_31762_20_9', '16_31762_20_9',
                                  '16_31762_20_9', '16_31762_20_9',
                                  '16_31762_20_9', '16_31762_20_9', '16_31762_20_9',
                                  '16_31762_20_9', '16_31762_20_9',
                                  '16_31762_20_9', '16_31762_20_9'],
                     'cell_id_list': [340, 360, 372, 400, 429,
                                      443, 574, 613, 624, 626,
                                      647, 648]}}

cluster_colors = {'cluster_1': [252 / 256, 103 / 256, 110 / 256],
                   'cluster_2': [106 / 255, 103 / 256, 206 / 256],
                    'cluster_3': [23 / 256, 157 / 256, 186 / 256],
                    'cluster_4': [55 / 256, 197 / 256, 151 / 256]}

# save cluster segmentations with nuclear overlay
for cluster in ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4']:
    crop_dir = os.path.join(plot_dir, 'Figure_5g_{}'.format(cluster))
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    current_fov_list, current_cell_id_list = cluster_info[cluster]['fov_list'], cluster_info[cluster]['cell_id_list']

    for i in range(len(current_fov_list)):
        current_fov = current_fov_list[i]
        current_cell_id = current_cell_id_list[i]
        idx = np.where(np.logical_and(cluster_df_selected_nucleated['point'] == current_fov,
                                      cluster_df_selected_nucleated['label'] == current_cell_id))[0][0]

        centroid_row, centroid_col, nuc_id = cluster_df_selected_nucleated.iloc[idx][['centroid-0', 'centroid-1', 'label_nuclear']]

        centroid_row = int(centroid_row)
        centroid_col = int(centroid_col)

        current_label = segmentation_labels_selected.loc[current_fov, :, :, 'whole_cell'].values
        offset = 40
        current_label_crop = current_label[centroid_row - offset: centroid_row + offset,
                                           centroid_col - offset: centroid_col + offset]
        mask = current_label_crop == current_cell_id
        segmentation_img = np.zeros((80, 80), dtype='uint8')
        segmentation_img[mask] = 255

        current_nuc_label = nuc_labels_selected.loc[current_fov, :, :, 'nuclear'].values
        offset = 40
        current_nuc_label_crop = current_nuc_label[centroid_row - offset: centroid_row + offset,
                             centroid_col - offset: centroid_col + offset]
        nuc_mask = current_nuc_label_crop == nuc_id
        nuc_segmentation_img = np.zeros((80, 80), dtype='uint8')
        nuc_segmentation_img[nuc_mask] = 255

        overlay_img = np.zeros((80, 80), dtype='uint8')
        overlay_img[mask] = 255
        overlay_img[nuc_mask] = 122

        # rgb_img = np.zeros((80, 80, 4))
        rgb_img = np.zeros((80, 80, 3), dtype='float')
        color_vec = cluster_colors[cluster]
        rgb_img[mask, :] = color_vec
        rgb_img[nuc_mask, :] = [0.7, 0.7, 0.7]

        io.imsave(os.path.join(crop_dir, 'segmentation_img_nuc_{}.png'.format(i)), nuc_segmentation_img)
        io.imsave(os.path.join(crop_dir, 'segmentation_img_{}.png'.format(i)), segmentation_img)
        io.imsave(os.path.join(crop_dir, 'segmentation_img_overlay_{}.png'.format(i)), overlay_img)
        io.imsave(os.path.join(crop_dir, 'segmentation_img_overlay_rgb_{}.png'.format(i)), rgb_img)

#Figure 5h
kmeans_cmap_vals = [[0, 0, 0],
                    [252 / 256, 103 / 256, 110 / 256],
                    [106 / 255, 103 / 256, 206 / 256],
                    [23 / 256, 157 / 256, 186 / 256],
                    [55 / 256, 197 / 256, 151 / 256]]

kmeans_cmap = ListedColormap(kmeans_cmap_vals)

example_fov = '6_31725_8_2'
current_label = segmentation_labels_selected.loc[example_fov, :, :, 'whole_cell'].values
current_cluster = cluster_df_selected.loc[cluster_df_selected['point'] == example_fov, :]
img = figures.label_image_by_value(current_label, current_cluster['label'].values, current_cluster['kmeans_labels'].values)


img_cm = kmeans_cmap(img / 4)
io.imsave(os.path.join(data_dir, 'plots/Figure_5h_example_early_cropped.png'), img_cm[800:2000, 800:2000])
io.imsave(os.path.join(data_dir, 'plots/Figure_5h_example_early.png'), img_cm)


# Figure 5i
example_fov = '16_31762_20_9'
current_label = segmentation_labels_selected.loc[example_fov, :, :, 'whole_cell'].values
current_cluster = cluster_df_selected.loc[cluster_df_selected['point'] == example_fov, :]
img = figures.label_image_by_value(current_label, current_cluster['label'].values, current_cluster['kmeans_labels'].values)

img_cm = kmeans_cmap(img / 4)
io.imsave(os.path.join(data_dir, 'plots/Figure_5i_example_late.png'), img_cm)
io.imsave(os.path.join(data_dir, 'plots/Figure_5i_example_late_cropped.png'), img_cm[600:1800, 600:1800])


rows = [[930, 1130], [1300, 1500], [1750, 1950]]
cols = [[960, 1160], [1500, 1700], [1300, 1500]]

for i in range(len(rows)):
    row_start, row_end = rows[i]
    col_start, col_end = cols[i]

    img = io.imread(os.path.join(data_dir, 'plots/Figure_5h_example_early.png'))
    io.imsave(os.path.join(data_dir, 'plots/Figure_5h_example_early_crop_{}.png'.format(i)),
              img[row_start:row_end, col_start:col_end, :])


# identify ROIs for kmeans overlay images

rows = [[700, 900], [1475, 1675], [750, 950]]
cols = [[720, 920], [1150, 1350], [1150, 1350]]

for i in range(len(rows)):
    row_start, row_end = rows[i]
    col_start, col_end = cols[i]

    img = io.imread(os.path.join(data_dir, 'plots/Figure_5i_example_late.png'))
    io.imsave(os.path.join(data_dir, 'plots/Figure_5i_example_late_crop_{}.png'.format(i)),
              img[row_start:row_end, col_start:col_end, :])


# Figure 5j
ratio_list, stage_list = [], []
for fov in selected_fovs:
    subset_df = cluster_df_selected.loc[cluster_df_selected['point'] == fov, :]
    subset_df = subset_df.loc[subset_df['nucleated'] == True, :]
    ratio = np.sum(subset_df['kmeans_labels'] == 2) / np.sum(subset_df['kmeans_labels'] == 1)
    ratio_list.append(ratio)
    stage_list.append(np.unique(subset_df['stage'])[0])

plot_df = pd.DataFrame({'ratio': ratio_list, 'stage': stage_list})
plot_df['ratio_log2'] = np.log2(plot_df['ratio'].values)

fig, ax = plt.subplots()
ax = sns.swarmplot(data=plot_df,  x='stage', y='ratio', color='blue')
# ax = sns.boxplot(data=plot_df,  x='stage', y='ratio', color='blue')
ax = sns.barplot(data=plot_df, x='stage', y='ratio', ci=None, color='grey')
ax.set_ylim((0, 4))
plt.savefig(os.path.join(plot_dir, 'Figure_5j.pdf'))
