import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import figures

import matplotlib
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Figure 1b, dataset size
base_dir = '/Users/noahgreenwald/Downloads/publications_data_folder/'
plot_dir = base_dir + 'plots'

published_data = pd.read_csv(os.path.join(base_dir, 'other', 'published_annotations.csv'))


tissue_nuc = np.sum(published_data.loc[published_data['image_type'] == 'tissue', :]['nuclear_annotations'].values)
tissue_cell = np.sum(published_data.loc[published_data['image_type'] == 'tissue', :]['cyto_annotations'])

culture_nuc = np.sum(published_data.loc[published_data['image_type'] == 'cell_culture', :]['nuclear_annotations'])
culture_cell = np.sum(published_data.loc[published_data['image_type'] == 'cell_culture', :]['cyto_annotations'])

previous_nuc = tissue_nuc + culture_nuc
previous_cell = tissue_cell + culture_cell

ours_nuc = 1100000
ours_cell = 1200000

summary = pd.DataFrame({'dataset': ['all', 'all', 'TissueNet', 'TissueNet'],
                        'compartment': ['nuc', 'cell', 'nuc', 'cell'],
                        'values': [np.sum(published_data['nuclear_annotations']),
                                   np.sum(published_data['cyto_annotations']),
                                   ours_nuc, ours_cell]})

g = sns.catplot(data=summary, kind='bar', x='compartment', y='values', color='blue', hue='dataset',
                hue_order=['all', 'TissueNet'])

plt.savefig(os.path.join(plot_dir, 'Figure_1b.pdf'))

# Figure 1c, annotations per platform
platform_counts = np.load(base_dir + 'other/platform_counts.npz',
                        allow_pickle=True)['stats'].item()


platform_counts = pd.DataFrame(platform_counts)
platform_counts = pd.DataFrame(platform_counts)
platform_vals = platform_counts.iloc[0, :].values
platform_names = platform_counts.columns.values

sort_idx = np.argsort(-platform_vals)
fig, ax = plt.subplots(figsize=(5, 5))
figures.barchart_helper(ax=ax, values=platform_vals[sort_idx],
                        labels=platform_names[sort_idx],
                        title='Cells per platform type', colors='blue')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, 'Figure_1c.pdf'))

# Figure 1d, annotations per organ
tissue_counts = np.load(base_dir + 'other/tissue_counts_detail.npz',
                        allow_pickle=True)['stats'].item()


tissue_counts = pd.DataFrame(tissue_counts)
tissue_vals = tissue_counts.iloc[0, :].values
tissue_names = tissue_counts.columns.values

sort_idx = np.argsort(-tissue_vals)
fig, ax = plt.subplots(figsize=(5, 5))
figures.barchart_helper(ax=ax, values=tissue_vals[sort_idx],
                        labels=tissue_names[sort_idx],
                        title='Cells per tissue type', colors='blue')
ax.set_ylim((0, 350000))
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, 'Figure_1d.pdf'))


# figure 1e, hours for construction
total_time = 4050
internal_hours = 160
fig, ax = plt.subplots(figsize=(3, 3))
figures.barchart_helper(ax=ax, values=[total_time, internal_hours],
                        labels=['Crowdsource', 'QC'],
                        title='Total hours',
                        colors='blue')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, 'Figure_1e.pdf'))
