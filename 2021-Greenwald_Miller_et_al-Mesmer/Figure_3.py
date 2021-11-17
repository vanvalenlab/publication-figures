import os
import shutil
import xarray as xr

import matplotlib.pyplot as plt

import skimage.io as io

import numpy as np
import pandas as pd
import seaborn as sns

from skimage.segmentation import find_boundaries
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from deepcell_toolbox.metrics import Metrics

import figures
from ark.utils import io_utils

import matplotlib
from scipy.stats import hmean

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Figure 3a overlays
base_dir = 'path_to_zip_folder/publications_data_folder/'
data_dir = os.path.join(base_dir, 'test_split_predictions')
plot_dir = base_dir + 'plots'

# Figure 3a
y_pred = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_512x512_predictions.npz'))['y']
true_dict = np.load(os.path.join(data_dir, '20201018_multiplex_final_seed_1_test_512x512.npz'))
X_true, y_true, tissue_list, platform_list = true_dict['X'], true_dict['y'], true_dict['tissue_list'], true_dict['platform_list']

indices =[330, 168, 132, 33]
name = ['Vectra_Breast_Cancer', 'MIBI_GI', 'CyCIF_Lung_Cancer', 'CODEX_Pancreas']

for i in range(len(indices)):
    idx = indices[i]
    pred_label = y_pred[idx, :, :, 0]
    true_label = y_true[idx, :, :, 0]
    DNA, Membrane = X_true[idx, :, :, 0], X_true[idx, :, :, 1]

    pred_bool = find_boundaries(pred_label, mode='inner')

    greyscale_cells = np.full((512, 512), 255, dtype='uint8')
    greyscale_cells[pred_label > 0] = 0
    greyscale_cells[pred_bool] = 160
    current_metrics = Metrics('test_split')

    true_label = label(true_label)
    pred_label = label(pred_label)

    current_metrics.calc_object_stats(y_true=np.expand_dims(np.expand_dims(true_label, axis=0), axis=-1),
                                      y_pred=np.expand_dims(np.expand_dims(pred_label, axis=0), axis=-1))
    current_preds = current_metrics.predictions[0][0]

    current_errors = np.full((512, 512), 255, dtype='uint8')
    current_errors[greyscale_cells == 0] = 160
    accurate_cells = current_preds['correct']['y_pred']
    accurate_mask = np.isin(pred_label, accurate_cells)
    accurate_mask[pred_bool] = False
    current_errors[accurate_mask] = 0

    crop_dir = os.path.join(plot_dir, 'Figure_3a_crop_{}'.format(i))
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    io.imsave(crop_dir + '/DNA.tiff', DNA)
    io.imsave(crop_dir + '/Membrane.tiff', Membrane)
    io.imsave(crop_dir + '/Greyscale.tiff', greyscale_cells)
    io.imsave(crop_dir + '/Errors.tiff', current_errors)
    io.imsave(crop_dir + '/Label.tiff', pred_label)

# specialist model evaluation
data_dir = base_dir + 'benchmarking_accuracy'

tissue_accuracy = np.load(os.path.join(data_dir, '20201018_tissue_accuracy_100.npz'), allow_pickle=True)
platform_accuracy = np.load(os.path.join(data_dir, '20201018_platform_accuracy_100.npz'), allow_pickle=True)

array_stack = []

# construct data from replicates for tissue models
tissue_list, model_list, f1_list = [], [], []
for split in ['1', '2', '3']:
    tissue_types = ['gi', 'breast', 'pancreas', 'immune', 'all']
    temp_dict = {}
    for tissue in tissue_types:
        vals = tissue_accuracy[split].item()[tissue]
        temp_dict[tissue] = vals


    tissue_array = figures.create_f1_score_grid(temp_dict, tissue_types)
    array_stack.append(tissue_array)
    tissue_list_temp, model_list_temp, f1_list_temp = figures.create_f1_score_long_df(data_array=tissue_array,
                                                                         unique_subsets=tissue_types)
    tissue_list.extend(tissue_list_temp)
    model_list.extend(model_list_temp)
    f1_list.extend(f1_list_temp)

array_vals = [x.values for x in array_stack]
array_vals = np.stack(array_vals, axis=0)
array_vals = np.mean(array_vals, axis=0)
tissue_array = pd.DataFrame(array_vals, columns=array_stack[0].columns, index=array_stack[0].index)

tissue_type_df = pd.DataFrame({'tissue_type': tissue_list,
              'model_type': model_list,
              'f1_score': f1_list})

# Figure 3b
fig, ax = plt.subplots()
ax = sns.barplot(data=tissue_type_df, x='tissue_type', y='f1_score', hue='model_type',
                 hue_order=['custom', 'general'],
                 color='grey', ci=None)
ax.get_legend().remove()
ax = sns.swarmplot(data=tissue_type_df, x='tissue_type', y='f1_score', hue='model_type',
                   hue_order=['custom', 'general'],
                   color='grey', dodge=True, alpha=0.5)
ax.get_legend().remove()
ax.set(ylim=(0, 1))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(plot_dir, 'Figure_3b.pdf'))

# compute p-vals for tissue
p_vals = []
for tissue in ['immune', 'pancreas', 'breast', 'gi']:

    mesmer_idx = np.logical_and(tissue_type_df['tissue_type'] == tissue,
                                tissue_type_df['model_type'] == 'general')

    specialist_idx = np.logical_and(tissue_type_df['tissue_type'] == tissue,
                                tissue_type_df['model_type'] == 'custom')

    _, p_val = ttest_ind(tissue_type_df.loc[mesmer_idx, 'f1_score'].values,
              tissue_type_df.loc[specialist_idx, 'f1_score'].values)
    p_vals.append(p_val)

# Figure S3a
g = sns.heatmap(data=tissue_array, annot=True, vmin=0, cmap='Blues')
plt.savefig(os.path.join(plot_dir, 'Figure_S3a.pdf'))

# construct data from replicates for platforms
array_stack = []
platform_list, model_list, f1_list = [], [], []
for split in ['1', '2', '3']:
    platform_types = ['vectra', 'mibi', 'cycif', 'codex', 'all']
    temp_dict = {}
    for platform in platform_types:
        vals = platform_accuracy[split].item()[platform]
        temp_dict[platform] = vals


    platform_array = figures.create_f1_score_grid(temp_dict, platform_types)
    array_stack.append(platform_array)
    platform_list_temp, model_list_temp, f1_list_temp = figures.create_f1_score_long_df(data_array=platform_array,
                                                                         unique_subsets=platform_types)
    platform_list.extend(platform_list_temp)
    model_list.extend(model_list_temp)
    f1_list.extend(f1_list_temp)

array_vals = [x.values for x in array_stack]
array_vals = np.stack(array_vals, axis=0)
array_vals = np.mean(array_vals, axis=0)
platform_array = pd.DataFrame(array_vals, columns=array_stack[0].columns, index=array_stack[0].index)

platform_type_df = pd.DataFrame({'platform_list': platform_list,
              'model_type': model_list,
              'f1_score': f1_list})

# Figure 3c
fig, ax = plt.subplots()
ax = sns.barplot(data=platform_type_df, x='platform_list', y='f1_score', hue='model_type',
                 hue_order=['custom', 'general'],
                 color='grey', ci=None)
ax.get_legend().remove()
ax = sns.swarmplot(data=platform_type_df, x='platform_list', y='f1_score', hue='model_type',
                   hue_order=['custom', 'general'],
                   color='grey', dodge=True, alpha=0.5)
ax.get_legend().remove()
ax.set(ylim=(0, 1))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(plot_dir, 'Figure_3c.pdf'))

# p vals for platform
for platform in ['mibi', 'vectra', 'cycif', 'codex']:

    mesmer_idx = np.logical_and(platform_type_df['platform_list'] == platform,
                                platform_type_df['model_type'] == 'general')

    specialist_idx = np.logical_and(platform_type_df['platform_list'] == platform,
                                    platform_type_df['model_type'] == 'custom')

    _, p_val = ttest_ind(tissue_type_df.loc[mesmer_idx, 'f1_score'].values,
              tissue_type_df.loc[specialist_idx, 'f1_score'].values)
    p_vals.append(p_val)

# Figure S3b
g = sns.heatmap(data=platform_array, annot=True, vmin=0, cmap='Blues')
plt.savefig(os.path.join(plot_dir, 'Figure_S3b.pdf'))


# create scalebar that goes from 0 to 1
tissue_array[0, 0] = 1
g = sns.heatmap(data=platform_array, annot=True, vmin=0, cmap='Blues')
plt.savefig(os.path.join(plot_dir, 'Figure_S3b_scalebar.pdf'))

# Figure 3d
data_dir = base_dir + 'Human_agreement/'
folders = ['DCIS_2328', 'Eliot_Point17', 'P101_T3_T4_Point2', 'cHL_Point8908']
folder_names = ['DCIS_MIBI', 'Colon_IF', 'Esophagus_MIBI', 'Hodgekins_Vectra']
f1_list, tissue_list, annotator_list = [], [], []

for i in range(len(folders)):
    # get all of the human annotations
    folder_path = os.path.join(data_dir, folders[i], 'annotations')
    img_names = io_utils.list_files(folder_path, '.tiff')
    imgs = []
    for img in img_names:
        current_img = io.imread(os.path.join(folder_path, img))
        imgs.append(current_img)
    f1_scores_human = figures.calculate_human_f1_scores(image_list=imgs)
    tissue_name = folder_names[i]
    f1_list.extend(f1_scores_human)
    tissue_list.extend([tissue_name] * len(f1_scores_human))
    annotator_list.extend(['human'] * len(f1_scores_human))

    # compare algorithm
    pred_img = io.imread(os.path.join(data_dir, folders[i], 'segmentation_label.tiff'))
    pred_img = np.expand_dims(pred_img, axis=0)
    f1_scores_alg = figures.calculate_alg_f1_scores(image_list=imgs, alg_pred=pred_img)

    f1_list.extend(f1_scores_alg)
    tissue_list.extend([tissue_name] * len(f1_scores_alg))
    annotator_list.extend(['alg'] * len(f1_scores_alg))


human_comparison_df = pd.DataFrame({'tissue': tissue_list, 'annotator_type': annotator_list,
                                    'f1': f1_list})

human_comparison_df.to_csv(os.path.join(base_dir + 'human_alg_scores.csv'))
human_comparison_df = pd.read_csv(os.path.join(data_dir, 'human_alg_scores.csv'))

# duplicate data to create an 'all' category
new_df = pd.DataFrame({'tissue': ['all'] * len(human_comparison_df),
                       'annotator_type': human_comparison_df['annotator_type'].values,
                       'f1': human_comparison_df['f1'].values})

human_comparison_df = human_comparison_df.append(new_df)


human_comparison_df['detailed'] = [human_comparison_df['tissue'].values[x] + human_comparison_df['annotator_type'].values[x] for x in range(len(human_comparison_df))]
g = sns.catplot(data=human_comparison_df,  x='detailed', y='f1', color='blue', kind='swarm')
g.set(ylim=(0, 1))

plt.savefig(os.path.join(plot_dir, 'Figure_3d.pdf'))

# compute p-vals
ttest_ind(human_comparison_df.loc[human_comparison_df['detailed'] == 'allalg', 'f1'].values,
          human_comparison_df.loc[human_comparison_df['detailed'] == 'allhuman', 'f1'].values)

# Figure 3f
data_dir = base_dir + 'Human_agreement/expert_evaluation'
eval_df = pd.DataFrame()
names = ['Albert_processed.csv', 'Eric_processed.csv', 'Hugo_processed.csv', 'saman_processed.csv']

for name in names:
    temp_df = pd.read_csv(os.path.join(data_dir, name))
    temp_df['expert'] = name
    eval_df = eval_df.append(temp_df)
fig, ax = plt.subplots()
ax = sns.swarmplot(data=eval_df,  x='img', y='rating')
ax = sns.boxplot(data=eval_df,  x='img', y='rating')

plt.savefig(os.path.join(plot_dir, 'Figure_3f.pdf'))

# compute p-values
ttest_1samp(eval_df.loc[eval_df['img'] == 'eliot', :]['rating'], 3, alternative='less')

data_dir = base_dir + 'benchmarking_accuracy/'
# Training dataset size benchmarking
model_list = []
split_list = []
f1_list = []
splits_list = []
merge_list = []
fp_list = []
fn_list = []
cat_list = []

for seed in [1, 2, 3]:
    f1_scores = np.load(os.path.join(data_dir, 'size_subset_metrics_seed_{}.npz'.format(seed)),
                        allow_pickle=True)
    data_splits = list(f1_scores.keys())

    for split in data_splits:
        current_slice = f1_scores[split].item()['tissue_stats']['all']
        current_f1 = current_slice['f1']
        current_split = current_slice['split']
        current_merge = current_slice['merge']
        current_cat = current_slice['catastrophe']
        current_fp = current_slice['gained_detections']
        current_fn = current_slice['missed_detections']
        f1_list.append(current_f1)
        splits_list.append(current_split)
        merge_list.append(current_merge)
        cat_list.append(current_cat)
        model_list.append('all')
        split_list.append(split)
        fp_list.append(current_fp)
        fn_list.append(current_fn)


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/labeled_data/'
dataset_array = pd.DataFrame({'model': model_list,
                              'f1': f1_list,
                              'split_error': splits_list,
                              'merge_error': merge_list,
                              'fp': fp_list,
                              'fn': fn_list,
                              'catastrophe': cat_list,
                              'split': split_list})
dataset_array.to_csv(os.path.join(data_dir, 'accuracy_by_dataset_size.csv'), index=False)

plots = ['catastrophe', 'split_error', 'merge_error', 'fp', 'fn']

for plot_type in plots:
    fig, ax = plt.subplots()
    ax = sns.swarmplot(data=dataset_array, x='split', y=plot_type, hue='model')

    ax = sns.pointplot(data=dataset_array, x='split', y=plot_type, hue='model',ci=None)
    ax.set(ylim=(0))

    plt.savefig(os.path.join(plot_dir, 'Figure_S3c_{}.pdf'.format(plot_type)))

# Figure S3i
data_dir = base_dir + 'test_split_predictions/'
true_dict = np.load(data_dir + '20201018_multiplex_final_seed_1_test_256x256.npz')
x_true, y_true, tissue_list, platform_list = true_dict['X'], true_dict['y'], true_dict['tissue_list'], true_dict['platform_list']
y_pred = np.load(data_dir + '20201018_multiplex_final_seed_1_test_cell_prediction.npz')['y']

m = Metrics('default_name')
m.calc_object_stats(y_true=y_true, y_pred=y_pred)

m.stats['recall'] = m.stats['correct_detections'] / m.stats['n_true']
m.stats['precision'] = m.stats['correct_detections'] / m.stats['n_pred']
m.stats['f1'] = hmean([m.stats['recall'], m.stats['precision']])
score_order = np.argsort(m.stats['f1']).values
x1 = x_true[score_order[0], :, :, 0].astype('float32')
io.imshow(x_true[score_order[24], :, :, 1].astype('float32'))

io.imsave(plot_dir + 'Figure_S3i_crop_0_DNA.tiff', x_true[540, :, :, 0].astype('float32'))
io.imsave(plot_dir + 'Figure_S3i_crop_0_Membrane.tiff', x_true[540, :, :, 1].astype('float32'))

io.imsave(plot_dir + 'Figure_S3i_crop_1_DNA.tiff', x_true[897, :, :, 0].astype('float32'))
io.imsave(plot_dir + 'Figure_S3i_crop_1_Membrane.tiff', x_true[897, :, :, 1].astype('float32'))

io.imsave(plot_dir + 'Figure_S3i_crop_2_DNA.tiff', x_true[300, :, :, 0].astype('float32'))
io.imsave(plot_dir + 'Figure_S3i_crop_2_Membrane.tiff', x_true[300, :, :, 1].astype('float32'))

io.imsave(plot_dir + 'Figure_S3i_crop_3_DNA.tiff', x_true[213, :, :, 0].astype('float32'))
io.imsave(plot_dir + 'Figure_S3i_crop_3_Membrane.tiff', x_true[213, :, :, 1].astype('float32'))

io.imsave(plot_dir + 'Figure_S3i_crop_4_DNA.tiff', x_true[697, :, :, 0].astype('float32'))
io.imsave(plot_dir + 'Figure_S3i_crop_4_Membrane.tiff', x_true[697, :, :, 1].astype('float32'))

# image distortion

# image blurring
blurred_f1, blurred_value = [], []
blurred_metrics_triplicate = np.load(os.path.join(data_dir, 'blurring_metrics_triplicate.npz'), allow_pickle=True)
for model in ['1', '2', '3']:
    blurred_metrics = blurred_metrics_triplicate[model].item()
    for key in blurred_metrics.keys():
        f1 = blurred_metrics[key]['all']['f1']
        blurred_f1.append(f1)
        blurred_value.append(key)

blurred_df = pd.DataFrame({'value': blurred_value, 'f1': blurred_f1})
fig, ax = plt.subplots()
ax = sns.swarmplot(data=blurred_df, x='value', y='f1',  color='blue')
ax = sns.pointplot(data=blurred_df, x='value', y='f1',  color='blue', ci=None)
ax.set(ylim=(0, 1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(os.path.join(plot_dir, 'Figure_S3j.pdf'))

# image resizing
resized_f1, resized_value = [], []
resized_metrics_triplicate = np.load(os.path.join(data_dir, 'resize_metrics_triplicate.npz'), allow_pickle=True)
for model in ['1', '2', '3']:
    resized_metrics = resized_metrics_triplicate[model].item()
    for key in resized_metrics.keys():
        f1 = resized_metrics[key]['all']['f1']
        resized_f1.append(f1)
        resized_value.append(key)

resized_df = pd.DataFrame({'value': resized_value, 'f1': resized_f1})

fig, ax = plt.subplots()
ax = sns.swarmplot(data=resized_df, x='value', y='f1',  color='blue')
ax = sns.pointplot(data=resized_df, x='value', y='f1',  color='blue', ci=None)
ax.set(ylim=(0, 1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(os.path.join(plot_dir, 'Figure_S3k.pdf'))

# image noising
noisy_f1, noisy_value = [], []
noisy_metrics_triplicate = np.load(os.path.join(data_dir, 'noise_metrics_triplicate.npz'), allow_pickle=True)
for model in ['1', '2', '3']:
    noisy_metrics = noisy_metrics_triplicate[model].item()
    for key in noisy_metrics.keys():
        f1 = noisy_metrics[key]['all']['f1']
        noisy_f1.append(f1)
        noisy_value.append(key)

noisy_df = pd.DataFrame({'value': noisy_value, 'f1': noisy_f1})

fig, ax = plt.subplots()
ax = sns.swarmplot(data=noisy_df, x='value', y='f1',  color='blue')
ax = sns.pointplot(data=noisy_df, x='value', y='f1',  color='blue', ci=None)
ax.set(ylim=(0, 1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(os.path.join(plot_dir, 'Figure_S3l.pdf'))