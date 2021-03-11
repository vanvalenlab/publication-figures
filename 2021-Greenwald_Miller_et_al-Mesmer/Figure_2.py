import os

import matplotlib.pyplot as plt
import skimage.io as io

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import hmean

from skimage.measure import regionprops_table
from skimage.measure import label

import figures

from deepcell_toolbox.metrics import Metrics
import matplotlib
from skimage.segmentation import find_boundaries

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

base_dir = 'path_to_zip_folder/publications_data_folder/'
plot_dir = base_dir + '/plots/'
benchmarking_dir = base_dir + '/benchmarking_accuracy/'

# create single df to hold the accuracy metrics for all models

files = ['retinamask_metrics_jacc.npz', 'nuclear_metrics_jacc.npz', 'MDC_metrics_jacc.npz',
         'featurenet_metrics_jacc.npz', 'cellpose_metrics_jacc.npz', 'featurenet_metrics_samir_jacc.npz',
         'stardist_metrics_jacc.npz']

names = ['RetinaMask', 'Mesmer_Nuclear', 'Mesmer', 'FeatureNet_Retrained', 'CellPose', 'FeatureNet',
         'StarDist']
metrics, scores, models = [], [], []
for i in range(len(files)):
    current_metrics = np.load(benchmarking_dir + files[i], allow_pickle=True)
    for split in ['1', '2', '3']:
        total_errors = 0
        for metric in ['f1', 'precision', 'recall', 'split', 'merge', 'catastrophe',
                       'gained_detections', 'missed_detections']:
            vals = current_metrics[split].item()
            metrics.append(metric)
            score = vals['tissue_stats']['all'][metric]
            scores.append(score)
            models.append(names[i])

            if metric in ['split', 'merge', 'catastrophe', 'gained_detections', 'missed_detections']:
                total_errors += score

        for metric in ['split', 'merge', 'catastrophe', 'gained_detections', 'missed_detections']:
            metrics.append(metric + '_normalized')
            score = vals['tissue_stats']['all'][metric] / total_errors
            scores.append(score)
            models.append(names[i])

        true_cells = vals['tissue_stats']['all']['n_true']
        for metric in ['split', 'merge', 'catastrophe', 'gained_detections', 'missed_detections']:
            metrics.append(metric + '_normalized_by_true_cells')
            score = vals['tissue_stats']['all'][metric] / true_cells
            scores.append(score)
            models.append(names[i])

        metrics.append('jaccard')
        scores.append(vals['jacc'])
        models.append(names[i])

        metrics.append('total_errors')
        scores.append(total_errors)
        models.append(names[i])

        metrics.append('accuracy_percent')
        scores.append(vals['tissue_stats']['all']['correct_detections'] / vals['tissue_stats']['all']['n_true'])
        models.append(names[i])

plotting_df = pd.DataFrame({'metric': metrics, 'score': scores, 'model': models})


model_speed = pd.read_csv(os.path.join(benchmarking_dir, 'f1_speed_dataframe.csv'))

# Figure 2b
plot_df = model_speed.loc[np.isin(model_speed['algorithm'].values, ['CellPose', 'Mesmer', 'FeatureNet']), :]
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.scatterplot(data=plot_df, x='speed', y='f1', hue='algorithm')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set(ylim=(0, 1))
ax.set(xlim=(0, 11))
plt.savefig(os.path.join(plot_dir, 'Figure_2b.pdf'))

# Figure S2b
model_stage = pd.read_csv(benchmarking_dir + 'model_time_by_stage.csv')
total_time = np.sum(model_stage['time'])
model_stage['percentage'] = model_stage['time'] / total_time
g = sns.catplot(data=model_stage, kind='bar', x='category', y='percentage')
plt.savefig(os.path.join(plot_dir, 'Figure_S2b.pdf'))

# Figure S2c
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.scatterplot(data=model_speed, x='speed', y='f1', hue='algorithm')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set(ylim=(0, 1))
ax.set(xlim=(0, 11))
plt.savefig(os.path.join(plot_dir, 'Figure_S2c.pdf'))

# Figure 2c
fig, ax = plt.subplots()
ax = sns.barplot(data=plotting_df, x='model', y='score', hue='metric',
                order=['Mesmer', 'FeatureNet', 'CellPose'],
                 hue_order=['precision', 'recall', 'jaccard'],
                 color='grey', ci=None)
ax.get_legend().remove()
ax = sns.swarmplot(data=plotting_df, x='model', y='score', hue='metric',
                   order=['Mesmer', 'FeatureNet', 'CellPose'],
                   hue_order=['precision', 'recall', 'jaccard'],
                   color='grey', dodge=True, alpha=0.5)
ax.get_legend().remove()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(plot_dir, 'Figure_2c.pdf'))

# Figure S2d
fig, ax = plt.subplots()
ax = sns.barplot(data=plotting_df, x='model', y='score', hue='metric',
                order=['Mesmer', 'Mesmer_Nuclear', 'FeatureNet_Retrained', 'RetinaMask', 'FeatureNet', 'StarDist', 'CellPose'],
                 hue_order=['precision', 'recall', 'jaccard'],
                 color='grey', ci=None)
ax.get_legend().remove()
ax = sns.swarmplot(data=plotting_df, x='model', y='score', hue='metric',
                   order=['Mesmer', 'Mesmer_Nuclear', 'FeatureNet_Retrained', 'RetinaMask',
                          'FeatureNet', 'StarDist', 'CellPose'],
                   hue_order=['precision', 'recall', 'jaccard'],
                   color='grey', dodge=True, alpha=0.5)
ax.get_legend().remove()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set(ylim=(0, 1))
plt.savefig(os.path.join(plot_dir, 'Figure_S2d.pdf'))


# Figure S2e
fig, ax = plt.subplots()
ax = sns.barplot(data=plotting_df, x='model', y='score', hue='metric',
                order=['Mesmer', 'Mesmer_Nuclear', 'FeatureNet_Retrained', 'RetinaMask', 'FeatureNet', 'StarDist', 'CellPose'],

                 hue_order=['missed_detections_normalized_by_true_cells', 'gained_detections_normalized_by_true_cells'],
                 color='grey', ci=None)

ax = sns.swarmplot(data=plotting_df, x='model', y='score', hue='metric',
                   order=['Mesmer', 'Mesmer_Nuclear', 'FeatureNet_Retrained', 'RetinaMask',
                          'FeatureNet', 'StarDist', 'CellPose'],
                   hue_order=['missed_detections_normalized_by_true_cells', 'gained_detections_normalized_by_true_cells'],
                   color='grey', dodge=True, alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(plot_dir, 'Figure_S2e_.pdf'))



# Figure S2f
fig, ax = plt.subplots()
ax = sns.barplot(data=plotting_df, x='model', y='score', hue='metric',
                order=['Mesmer', 'Mesmer_Nuclear', 'FeatureNet_Retrained', 'RetinaMask', 'FeatureNet', 'StarDist', 'CellPose'],

                 hue_order=['merge_normalized_by_true_cells', 'split_normalized_by_true_cells', 'catastrophe_normalized_by_true_cells'],
                 color='grey', ci=None)

ax = sns.swarmplot(data=plotting_df, x='model', y='score', hue='metric',
                   order=['Mesmer', 'Mesmer_Nuclear', 'FeatureNet_Retrained', 'RetinaMask',
                          'FeatureNet', 'StarDist', 'CellPose'],
                   hue_order=['merge_normalized_by_true_cells', 'split_normalized_by_true_cells', 'catastrophe_normalized_by_true_cells'],
                   color='grey', dodge=True, alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(plot_dir, 'Figure_S2f.pdf'))

# Figure S2g
table_models = ['Mesmer', 'CellPose', 'FeatureNet']
table_vals = ['precision', 'recall', 'f1', 'jaccard']

model_vals = []
for model in table_models:
    current_vals = []
    for metric in table_vals:
        idx = np.logical_and(plotting_df['model'] == model,
                             plotting_df['metric'] == metric)
        current_counts = plotting_df.loc[idx, 'score'].values
        current_vals.append(np.mean(current_counts))
        current_vals.append(np.std(current_counts))
    model_vals.append(current_vals)

model_vals = np.stack(model_vals, axis=0)
model_vals = np.round(model_vals, 4)

column_names = [[x, x + '_sd'] for x in table_vals]
column_names = np.concatenate(column_names, axis=0)

model_df = pd.DataFrame(model_vals, index=table_models, columns=column_names)
model_df.to_csv(plot_dir + '/Figure_S2g.csv')


# Figure 2f
data_dir = base_dir + 'image_files/'

true_labels = np.load(data_dir + 'fig2_true_labels.npz')
true_labels = true_labels['y'][0, :, :, 0]

# Figure 2f True
disp_img_true = figures.label_image_by_ratio(true_labels, true_labels)
disp_img_true_final = figures.apply_colormap_to_img(disp_img_true)
io.imsave(data_dir + 'Figure_2f_true.png', disp_img_true_final.astype('float32'))

# Figure 2f Mesmer
pred_labels_mesmer = io.imread(data_dir + 'Mesmer_prediction.tif')[:1008, :1008]
disp_img_mesmer = figures.label_image_by_ratio(true_labels, pred_labels_mesmer)
disp_img_mesmer_final = figures.apply_colormap_to_img(disp_img_mesmer)
io.imsave(data_dir + 'Figure_2f_mesmer.png', disp_img_mesmer_final.astype('float32'))

m = Metrics('test_split')
m.calc_object_stats(y_true=np.expand_dims(np.expand_dims(true_labels, axis=0), axis=-1),
                    y_pred=np.expand_dims(np.expand_dims(pred_labels_mesmer, axis=0), axis=-1))
m.stats['f1']


# Figure 2e crops
grey_seg_label = np.full(pred_labels_mesmer.shape, 255, dtype='uint8')
seg_boundaries = find_boundaries(pred_labels_mesmer, mode='inner')
grey_seg_label[pred_labels_mesmer > 0] = 0
grey_seg_label[seg_boundaries] = 160

grey_true_label = np.full(true_labels.shape, 255, dtype='uint8')
true_boundaries = find_boundaries(true_labels, mode='inner')
grey_true_label[true_labels > 0] = 0
grey_true_label[true_boundaries] = 160

io.imsave(plot_dir + 'Figure_2e_top.tiff', grey_true_label[700:800, 50:350])
io.imsave(plot_dir + 'Figure_2e_mid.tiff', grey_seg_label[700:800, 50:350])
io.imsave(plot_dir + 'Figure_2e_bottom.tiff', disp_img_mesmer_final[700:800, 50:350].astype('float32'))

fig, ax = plt.subplots()
pos = ax.imshow(disp_img_mesmer_final, cmap='coolwarm')
fig.colorbar(pos)
plt.savefig(os.path.join(data_dir, 'Figure_2e_colorbar.pdf'))

# Figure 2f Featurenet
pred_labels = io.imread(data_dir + 'featurenet_predictions.tif')
disp_img = figures.label_image_by_ratio(true_labels, pred_labels)
disp_img_final = figures.apply_colormap_to_img(disp_img)
io.imsave(plot_dir + 'Figure_2f_featurenet.png', disp_img_final.astype('float32'))

# compute f1
m = Metrics('test_split')
m.calc_object_stats(y_true=np.expand_dims(np.expand_dims(true_labels, axis=0), axis=-1),
                    y_pred=np.expand_dims(np.expand_dims(pred_labels, axis=0), axis=-1))
m.stats['f1']

# Figure 2f Cellpose
pred_labels = io.imread(data_dir + 'cellpose_predictions.tif')
disp_img = figures.label_image_by_ratio(true_labels, pred_labels)
disp_img_final = figures.apply_colormap_to_img(disp_img)
io.imsave(data_dir + 'Figure_2f_cellpose.png', disp_img_final.astype('float32'))

# compute f1
m = Metrics('test_split')
m.calc_object_stats(y_true=np.expand_dims(np.expand_dims(true_labels, axis=0), axis=-1),
                    y_pred=np.expand_dims(np.expand_dims(pred_labels, axis=0), axis=-1))
m.stats['f1']

# note: this code requires TissueNet to run, which will be made available following publication

# Figure 2g
# y_pred = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_512x512_predictions.npz'))['y']
# true_dict = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_512x512.npz'))
# X_true, y_true, tissue_list, platform_list = true_dict['X'], true_dict['y'], true_dict['tissue_list'], true_dict['platform_list']

# selected_indices = [328, 330, 79, 191, 252,246, 247,249,
#                     231, 232, 230, 172, 168, 182, 181,
#                     110, 132,222, 227, 237, 240, 33, 45,
#                     221, 135]
# selected_names = ['breast_cancer', 'breast_cancer', 'colon', 'crc', 'dcis', 'dcis', 'dcis', 'dcis',
#                   'epidermis', 'epidermis', 'epidermis', 'esophagus', 'esophagus', 'hiv_ln', 'hiv_ln',
#                   'lung_cancer', 'lung_cancer', 'lymphoma', 'lymphoma', 'melanoma', 'melanoma', 'pancreas', 'pancreas',
#                   'spleen', 'tb']
#
# final_indices = [328, 79, 191, 252,
#                     232, 172, 182,
#                     110, 227, 45,
#                     221, 135]
# final_names = ['breast_cancer', 'colon', 'crc', 'dcis',
#                   'epidermis', 'esophagus', 'hiv_ln',
#                   'lung_cancer', 'lymphoma', 'pancreas',
#                   'spleen', 'tb']
#
# for idx in range(len(final_indices)):
#     i = final_indices[idx]
#     true_label, pred_label = y_true[i, ..., 0], y_pred[i, ..., 0]
#     disp_img_cell = figures.label_image_by_ratio(true_label, pred_label)
#     disp_img_cell_final = figures.apply_colormap_to_img(disp_img_cell)
#     name = final_names[idx]
#     io.imsave(plot_dir + 'Figure_2_g_{}_{}.png'.format(idx, name), disp_img_cell_final.astype('float32'))
#
# # f1 scores for each image
# m = Metrics('test_split')
# m.calc_object_stats(y_true=y_true[final_indices, ...],
#                     y_pred=y_pred[final_indices, ...])
#
# m.stats['recall'] = m.stats['correct_detections'] / m.stats['n_true']
# m.stats['precision'] = m.stats['correct_detections'] / m.stats['n_pred']
# m.stats['f1'] = [hmean([m.stats['recall'].values[i], m.stats['precision'].values[i]]) for i in range(len(m.stats))]

# Figure S2h
# data_dir = base_dir + '/image_files/test_split_predictions/'
# true_dict = np.load(data_dir + '20201018_multiplex_final_seed_1_nuclear_test_256x256.npz')
# true_cell_labels = true_dict['y'][..., :1].astype('int16')
# true_x_data = true_dict['X']
# pred_cell_labels = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_cell_prediction.npz'))['y']
# pred_nuc_expansion_labels = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_nuc_expansion_prediction.npz'))['y']
#
# for i in range(true_cell_labels.shape[0]):
#     img = true_cell_labels[i, :, :, 0]
#     img = label(img)
#     true_cell_labels[i, :, :, 0] = img
#
# for i in range(pred_cell_labels.shape[0]):
#     img = pred_cell_labels[i, :, :, 0]
#     img = label(img)
#     pred_cell_labels[i, :, :, 0] = img
#
# for i in range(pred_nuc_expansion_labels.shape[0]):
#     img = pred_nuc_expansion_labels[i, :, :, 0]
#     img = label(img)
#     pred_nuc_expansion_labels[i, :, :, 0] = img
#
#
# from deepcell_toolbox.metrics import Metrics
# m = Metrics('whole_cell')
# m.calc_object_stats(y_true=true_cell_labels, y_pred=pred_cell_labels)
#
# jacc = figures.calc_jaccard_index_object(m.predictions, true_cell_labels, pred_cell_labels)
# x = np.concatenate(jacc)
# np.mean(x)
# np.savez_compressed(os.path.join(data_dir, 'cell_accuracy_metrics.npz'), metrics=m.predictions)
#
# m_nuc = Metrics('nuc')
# m_nuc.calc_object_stats(y_true=true_cell_labels, y_pred=pred_nuc_expansion_labels)
# np.savez_compressed(os.path.join(data_dir, 'nuc_expansion_accuracy_metrics.npz'), metrics=m_nuc.predictions)
#
# cell_preds = np.load(os.path.join(data_dir, 'cell_accuracy_metrics.npz'), allow_pickle=True)['metrics']
# nuc_exp_preds = np.load(os.path.join(data_dir, 'nuc_expansion_accuracy_metrics.npz'), allow_pickle=True)['metrics']
#
# prop_df_cell = pd.DataFrame()
# prop_df_nuc = pd.DataFrame()
# properties = ['label', 'area', 'eccentricity', 'major_axis_length', 'minor_axis_length',
#              'perimeter', 'centroid', 'convex_area',
#              'equivalent_diameter']
#
# for i in range(len(cell_preds)):
#     pred_label = pred_cell_labels[i, :, :, 0]
#     pred_expansion_label = pred_nuc_expansion_labels[i, :, :, 0]
#     true_label = true_cell_labels[i, :, :, 0]
#
#     # get ids of true and predicted cells which metrics evaluates as being correctly identified
#     good_ids_true = cell_preds[i][0]['correct']['y_true']
#     good_ids_pred = cell_preds[i][0]['correct']['y_pred']
#
#     true_props_table = pd.DataFrame(regionprops_table(true_label, properties=properties))
#     pred_props_table = pd.DataFrame(regionprops_table(pred_label, properties=properties))
#
#     # extract regionprops information from accurately matched cells and combine into single df
#     paired_df_cell = figures.get_paired_metrics(true_ids=good_ids_true, pred_ids=good_ids_pred,
#                                            true_metrics=true_props_table,
#                                            pred_metrics=pred_props_table)
#     paired_df_cell['img_num'] = i
#     prop_df_cell = prop_df_cell.append(paired_df_cell)
#
#     # same thing for nuclear expansion predictions
#     # get ids of true and predicted cells which metrics evaluates as being correctly identified
#     good_ids_true_nuc = nuc_exp_preds[i][0]['correct']['y_true']
#     good_ids_pred_nuc = nuc_exp_preds[i][0]['correct']['y_pred']
#
#     true_props_table_nuc = pd.DataFrame(regionprops_table(true_label, properties=properties))
#     pred_props_table_nuc = pd.DataFrame(regionprops_table(pred_expansion_label, properties=properties))
#
#     # extract regionprops information from accurately matched cells and combine into single df
#     paired_df_nuc = figures.get_paired_metrics(true_ids=good_ids_true_nuc, pred_ids=good_ids_pred_nuc,
#                                                 true_metrics=true_props_table_nuc,
#                                                 pred_metrics=pred_props_table_nuc)
#     paired_df_nuc['img_num'] = i
#     prop_df_nuc = prop_df_nuc.append(paired_df_nuc)
#
#
# bins = np.percentile(prop_df_cell['area_true'].values, np.arange(0, 101, 10))
#
# prop_df_cell['bin'] = 9
# prop_df_nuc['bin'] = 9
# for i in range(10):
#     bin_start = bins[i]
#     bin_end = bins[i + 1]
#     idx = np.logical_and(prop_df_cell['area_true'] > bin_start,
#                          prop_df_cell['area_true'] <= bin_end)
#     prop_df_cell.loc[idx, 'bin'] = i
#
#     idx = np.logical_and(prop_df_nuc['area_true'] > bin_start,
#                          prop_df_nuc['area_true'] <= bin_end)
#     prop_df_nuc.loc[idx, 'bin'] = i
#
# prop_df_cell['pred_log2'] = np.log2(prop_df_cell['area_pred'].values / prop_df_cell['area_true'].values)
# prop_df_nuc['pred_log2'] = np.log2(prop_df_nuc['area_pred'].values / prop_df_nuc['area_true'].values)
#
# prop_df_cell.to_csv(os.path.join(data_dir, 'cell_properties.csv'), index=False)
# prop_df_nuc.to_csv(os.path.join(data_dir, 'nuc_properties.csv'), index=False)
#
# prop_df_cell = pd.read_csv(os.path.join(data_dir, 'cell_properties.csv'))
# prop_df_nuc = pd.read_csv(os.path.join(data_dir, 'nuc_properties.csv'))
#
# g = sns.catplot(data=prop_df_cell,
#                 kind='violin', x='bin', y='pred_log2', showfliers=True,
#                 order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], color='grey')
#
# g.set(ylim=(-1, 1))
# g.fig.set_size_inches(14, 5)
#
# plt.savefig(os.path.join(plot_dir, 'Figure_S2h_left.pdf'))
#
# g = sns.catplot(data=prop_df_nuc,
#                 kind='violin', x='bin', y='pred_log2', showfliers=True,
#                 order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], color='grey')
#
# g.set(ylim=(-1, 1))
# g.fig.set_size_inches(14, 5)
#
# plt.savefig(os.path.join(plot_dir, 'Figure_S2h_right.pdf'))
