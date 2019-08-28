# figures.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .utils import MissingDataError

class BaseFigure(object):

    def __init__(self, raw_data, chosen_delay, chosen_img_num):
        self.raw_data = raw_data
        self.chosen_delay = chosen_delay
        self.chosen_img_num = chosen_img_num

    def format_data_for_error_plotting(self, data_lists):
        means = []
        std_errs_of_means = []
        percentiles_975 = []
        percentiles_025 = []
        one_sided_95_percent_intervals = []
        for datum_list in data_lists:
            # compute mean
            total = 0
            for datum in datum_list:
                total = total + datum
            mean = total/len(datum_list)
            means.append(mean)

            # compute standard error of mean
            squared_deviations = 0
            for datum in datum_list:
                squared_deviations = squared_deviations + (datum - mean)**2
            std_dev = ( (1/(len(datum_list)-1)) * squared_deviations)**(1/2)
            std_err_of_mean = std_dev/(len(datum_list)**(1/2))
            std_errs_of_means.append(std_err_of_mean)

            # compute 95% confidence intervals
            percentile_975 = mean + (std_err_of_mean * 1.96)
            percentile_025 = mean - (std_err_of_mean * 1.96)
            percentiles_975.append(percentile_975)
            percentiles_025.append(percentile_025)
            one_sided_95_percent_intervals.append(std_err_of_mean * 1.96)
        return (means, one_sided_95_percent_intervals)

    def define_colorblind_color_map(self):
        # (R,G,B) from nature methods Sky blue, bluish green, reddish purple, vermillion, orange
        colors = [(86/255,180/255,233/255), (0,158/255,115/255), (204/255,121/255,167/255), (213/255,94/255,0), (230/255,159/255,0)]

        n_bins = 5
        cmap_name = 'colorblind'
        self.cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


class ImageTimeVsGpu(BaseFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, title_label, pdf_label):
        super().__init__(raw_data, chosen_delay, chosen_img_num)
        self.plot_title = title_label + " Image Semantic Segmentation Runtime (Per Image)"
        self.plot_pdf_name = pdf_label + "_image_runtimes.pdf"
        self.y_label = "Time (Minutes)"

    def refine_data(self):
        # only choose relevant runs
        refined_data = []
        for entry in self.raw_data:
            if entry["start_delay"] == self.chosen_delay:
                refined_data.append(entry)
        refined_data2 = []
        for entry in refined_data:
            if entry["num_images"] == self.chosen_img_num:
                refined_data2.append(entry)
        if len(refined_data2) == 0:
            raise MissingDataError
        
        # grab variables of interest from relevant runs
        variables_of_interest = ['average_image_time', 'average_image_upload_time', 'average_image_prediction_time', 'average_image_postprocess_time', 'average_image_download_time']
        gpu_nums = [1,4,8]
        output_data = {}
        for variable_of_interest in variables_of_interest:
            data_lists = []
            for gpu_num in gpu_nums:
                times = [entry[variable_of_interest] for entry in refined_data2 if entry['num_gpus']==gpu_num]
                data_lists.append(times)
                output_data[variable_of_interest] = self.format_data_for_error_plotting(data_lists)
        
        # create DataFrame from variables of interest
        # 2 is a magic number that derives from the format of the data returned from self.format_data_for_error_plotting
        col_num = len(variables_of_interest)*2
        row_num = len(gpu_nums)
        data_array = np.zeros( (row_num, col_num) )
        for row in range(row_num):
            for col in range(col_num):
                var_num = col % len(variables_of_interest)
                var_index = int( (col - var_num) / len(variables_of_interest) )
                data_array[row,col] = output_data[ variables_of_interest[var_num] ][ var_index ][ row ]
        self.data_df = pd.DataFrame(data_array, columns=["total elapsed time","upload time","prediction time","postprocessing time","download time","total_err","upload_err","prediction_err","postprocessing_err","download_err"], index=["1GPU","4GPU","8GPU"])


    def plot(self):
        # preliminaries
        self.refine_data()
        self.define_colorblind_color_map()
        
        # plot and save figure
        n_col = len(self.data_df.columns) 
        fig, axes = plt.subplots(1, figsize=(20,10))
        self.axe = axes
        # make bar plots
        self.axe = self.data_df[["upload time", "prediction time", "postprocessing time", "download time"]].plot(kind="bar",
                        yerr=self.data_df[["upload_err", "prediction_err", "postprocessing_err", "download_err"]].values.T,
                        linewidth=0,
                        stacked=True,
                        ax=self.axe,
                        legend=False,
                        grid=False,
                        cmap=self.cmap)

        self.axe.set_title(self.plot_title)
        self.axe.set_ylabel(self.y_label)
        self.axe.set_xlabel('Processing time')
        self.axe.set_xticks([0,1,2])
        self.axe.set_xlim(-0.30,2.30)
        self.axe.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
        h,l = self.axe.get_legend_handles_labels() # get the handles we want to modify
        l1 = self.axe.legend(h[n_col-1::-1], l[n_col-1::-1], loc=[.425, 0.75])
        self.axe.add_artist(l1)
        plt.savefig(self.plot_pdf_name, transparent=True)

