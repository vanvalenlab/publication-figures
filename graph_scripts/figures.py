# figures.py

import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .utils import MissingDataError

from scipy import stats

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

    def define_colorblind_color_maps(self):
        # (R,G,B) from nature methods Sky blue, bluish green, reddish purple, vermillion, orange
        colors = [(86/255,180/255,233/255), (0,158/255,115/255), (204/255,121/255,167/255), (213/255,94/255,0), (230/255,159/255,0)]
        self.color_maps = []
        n_bins = 5
        for cut_point in range(len(colors)):
            cmap_name = 'colorblind' + str(cut_point+1)
            unflattened_new_color_list = [ colors[cut_point:] + colors[:cut_point] ]
            new_color_list = [color for color_list in unflattened_new_color_list for color in color_list]
            cmap = LinearSegmentedColormap.from_list(cmap_name, new_color_list, N=n_bins)
            self.color_maps.append(cmap)


class ImageTimeVsGpu(BaseFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, title_label, pdf_label):
        super().__init__(raw_data, chosen_delay, chosen_img_num)
        self.plot_title = "Semantic Segmentation Workflow Component Runtime"
        if chosen_delay==5.0:
            self.plot_pdf_name = pdf_label + "_5sdelay_image_runtimes.pdf"
        elif chosen_delay==0.5:
            self.plot_pdf_name = pdf_label + "_0point5sdelay_image_runtimes.pdf"
        self.y_label = "Density (unitless)"

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
        
        # create DataFrame from variables of interest in relevant runs
        variables_of_interest = ['all_total_times', 'all_network_times', 'all_upload_times', 'all_prediction_times', 'all_postprocess_times', 'all_download_times']
        gpu_nums = [1,4,8]
        replicates = 3
        variable_lengths = []
        for variable_of_interest in variables_of_interest:
            for gpu_num in gpu_nums:
                run_times = [entry[variable_of_interest] for entry in refined_data2 if entry['num_gpus']==gpu_num]
                for run_index, run in enumerate(run_times):
                    variable_lengths.append(len(run))
                    #if len(run) < max(variable_lengths):
                    #    print(f"Variable: {variable_of_interest}, gpu: {gpu_num}, replicate: {run_index}, length: {len(run)}")
        #if min(variable_lengths) != max(variable_lengths):
        #    print(variable_lengths)
        #    raise ValueError("I don't know how to deal properly with different list lengths.")
        row_num = max(variable_lengths)*replicates
        col_num = len(gpu_nums)*len(variables_of_interest)
        
        data_array = np.zeros( (row_num, col_num) )
        for col in range(col_num):
            variable_name = variables_of_interest[ col % len(variables_of_interest) ]
            gpu_number = gpu_nums[ int( col/len(variables_of_interest) ) % len(gpu_nums) ]
            relevant_data = [ entry[variable_name] for entry in refined_data2 if entry['num_gpus']==gpu_number ]
            #print(f"col: {col}, len(variables_of_interest): {len(variables_of_interest)}, variable_name: {variable_name}")
            #print(f"col: {col}, gpu_number: {gpu_number}")
            for row in range(row_num):
                replicate_number = int( row / max(variable_lengths) )
                replicate_length = len(relevant_data[replicate_number])
                replicate_desired_entry = row % max(variable_lengths)
                if replicate_desired_entry < replicate_length:
                    data_array[row,col] = relevant_data[replicate_number][replicate_desired_entry]
                else:
                    data_array[row,col] = np.nan
        self.data_df = pd.DataFrame(data_array, columns=[
            "1GPU_total",
            "1GPU_network",
            "1GPU_upload",
            "1GPU_prediction",
            "1GPU_postprocess",
            "1GPU_download",
            "4GPU_total",
            "4GPU_network",
            "4GPU_upload",
            "4GPU_prediction",
            "4GPU_postprocess",
            "4GPU_download",
            "8GPU_total",
            "8GPU_network",
            "8GPU_upload",
            "8GPU_prediction",
            "8GPU_postprocess",
            "8GPU_download"
        ])


    def plot(self):
        # preliminaries
        self.refine_data()
        self.define_colorblind_color_maps()
        
        # plot and save figures for each number of GPUs
        bin_number = 10000
        gpus = ["1GPU", "4GPU", "8GPU"]
        for gpu in gpus:
            n_col = len(self.data_df.columns) 
            if self.chosen_img_num == 1000000:
                if gpu == "1GPU":
                    xmaxes = [7,25,3,7]
                elif gpu == "4GPU":
                    xmaxes = [2,120,3,2]
                elif gpu == "8GPU":
                    xmaxes = [2,120,3,2]
            elif self.chosen_img_num == 100000:
                if gpu == "1GPU":
                    xmaxes = [7,25,3,7]
                elif gpu == "4GPU":
                    xmaxes = [2,120,3,2]
                elif gpu == "8GPU":
                    xmaxes = [2,120,3,2]
            elif self.chosen_img_num == 10000:
                if gpu == "1GPU":
                    xmaxes = [2,25,3,1.5]
                elif gpu == "4GPU":
                    xmaxes = [2,120,3,1.5]
                elif gpu == "8GPU":
                    xmaxes = [2,120,3,1.5]
            # choose type of time measurement for each subplot
            times = ["network", "prediction", "postprocess"]
            number_of_subplots = len(times)

            # Dummy figure and bin width computation
            # This block serves two purposes:
            # 1) We need to create a dummy figure so that we can grab the y-axis labels off of it,
            #    so that we can then modify those labels and use them on the real figure in the next block.
            # 2) We need to compute the bin widths before rewriting the y-axis labels.
            bin_widths = []
            if number_of_subplots == 4:
                fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
            elif number_of_subplots == 3:
                fig, self.axes = plt.subplots(nrows=1, ncols=3, figsize=(20,10))
            figure_title = "Semantic Segmentation Workflow Runtimes"
            fig.suptitle(figure_title, fontsize="x-large")
            for i in range(number_of_subplots):
                # compute bin width
                name = gpu + "_" + times[i]
                data_min = np.min(self.data_df[[name]])
                data_max = np.max(self.data_df[[name]])
                bin_width = (data_max - data_min)/bin_number
                bin_widths.append(bin_width)
                print(f"bin_width for {name}: {bin_width}")
                # plot dummy plot
                if number_of_subplots==4:
                    subplot_x = int("{:02b}".format(i)[0])
                    subplot_y = int("{:02b}".format(i)[1])
                    print(f"Creating plot {name} in position {subplot_x},{subplot_y}.")
                elif number_of_subplots==3:
                    print(f"Creating plot {name} in position {i}.")
                if number_of_subplots==4:
                    self.axe = self.data_df[[ name ]].plot.hist(
                        ax=self.axes[subplot_x,subplot_y],
                        bins=bin_number,
                        legend=False,
                        cmap=self.color_maps[i])
                elif number_of_subplots==3:
                    self.axe = self.data_df[[ name ]].plot.hist(
                        ax=self.axes[i],
                        bins=bin_number,
                        legend=False,
                        cmap=self.color_maps[i])
                if "postprocess" in times[i]:
                    base = times[i].capitalize()
                    suffix = "ing"
                    second_word = " Runtime"
                elif "prediction" in times[i]:
                    base = "Tensorflow Serving Response Time"
                    suffix = ""
                    second_word = ""
                elif "network" in times[i]:
                    base = "Data Transfer"
                    suffix = ""
                    second_word = " Time"
                plot_subtitle = base + suffix + second_word
                self.axe.set_title(plot_subtitle)
                self.axe.set_ylabel(self.y_label)
                self.axe.set_xlabel('Time (s)')
                self.axe.set_ylim(ymin=0)
                self.axe.set_xlim(xmin=0,xmax=xmaxes[i])
            gpu_plot_name = gpu + "_" + self.plot_pdf_name
            plt.savefig(gpu_plot_name, transparent=True)

            # get y axis labels from dummy figure
            labels = []
            label_maxes = []
            for i in range(number_of_subplots):
                subplot_labels = [ float(label.get_text()) for label in self.axes[i].get_yticklabels() ]
                try:
                    labels_max = np.max(subplot_labels)
                    label_maxes.append(labels_max)
                except TypeError:
                    pdb.set_trace()
                try:
                    modified_subplot_labels = [ str(label/labels_max)[:4] for label in subplot_labels ]
                except IndexError:
                    pdb.set_trace()
                labels.append(modified_subplot_labels)

            # Now, make real figure
            if number_of_subplots == 4:
                fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
            elif number_of_subplots == 3:
                fig, self.axes = plt.subplots(nrows=1, ncols=3, figsize=(20,10))
            figure_title = "Semantic Segmentation Workflow Runtimes"
            fig.suptitle(figure_title, fontsize="x-large")
            for i in range(number_of_subplots): # magic number for 2x2 plot
                name = gpu + "_" + times[i]
                if number_of_subplots==4:
                    subplot_x = int("{:02b}".format(i)[0])
                    subplot_y = int("{:02b}".format(i)[1])
                    print(f"Creating plot {name} in position {subplot_x},{subplot_y}.")
                elif number_of_subplots==3:
                    print(f"Creating plot {name} in position {i}.")
                # actually plot
                if number_of_subplots==4:
                    self.axe = self.data_df[[ name ]].plot.hist(
                        ax=self.axes[subplot_x,subplot_y],
                        bins=bin_number,
                        legend=False,
                        cmap=self.color_maps[i])
                elif number_of_subplots==3:
                    self.axe = self.data_df[[ name ]].plot.hist(
                        ax=self.axes[i],
                        bins=bin_number,
                        legend=False,
                        cmap=self.color_maps[i])
                if "postprocess" in times[i]:
                    base = times[i].capitalize()
                    suffix = "ing"
                    second_word = " Runtime"
                elif "prediction" in times[i]:
                    base = "Tensorflow Serving Response Time"
                    suffix = ""
                    second_word = ""
                elif "network" in times[i]:
                    base = "Data Transfer"
                    suffix = ""
                    second_word = " Time"
                plot_subtitle = base + suffix + second_word
                self.axe.set_title(plot_subtitle)
                self.axe.set_ylabel(self.y_label)
                self.axe.set_xlabel('Time (s)')
                self.axe.set_ylim(ymin=0,ymax=label_maxes[i]*1.01)
                self.axe.set_xlim(xmin=0,xmax=xmaxes[i])
                self.axe.set_yticklabels(labels[i])
            gpu_plot_name = gpu + "_" + self.plot_pdf_name
            plt.savefig(gpu_plot_name, transparent=True)

        if False:
            pdb.set_trace()
