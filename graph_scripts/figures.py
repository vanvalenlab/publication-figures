# figures.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path
from matplotlib.colors import LinearSegmentedColormap

from .utils import MissingDataError

class BaseFigure(object):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, output_folder):
        self.raw_data = raw_data
        self.chosen_delay = chosen_delay
        self.chosen_img_num = chosen_img_num
        self.chosen_img_nums = self.chosen_img_num # alias for figures using multiple image numbers
        self.output_folder = output_folder

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


class ImageTimeVsGpuFigure(BaseFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, pdf_label, output_folder):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder)
        self.plot_title = "Semantic Segmentation Workflow Component Runtime"
        if chosen_delay==5.0:
            self.plot_pdf_name = pdf_label + "_5sdelay_image_runtimes.pdf"
        elif chosen_delay==0.5:
            self.plot_pdf_name = pdf_label + "_0point5sdelay_image_runtimes.pdf"
        self.y_label = "Counts"

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

        # TODO: simplify series padding, which occurs here and in one block in graph_creation.py
        data_array = np.zeros( (row_num, col_num) )
        for col in range(col_num):
            variable_name = variables_of_interest[ col % len(variables_of_interest) ]
            gpu_number = gpu_nums[ int( col/len(variables_of_interest) ) % len(gpu_nums) ]
            relevant_data = [ entry[variable_name] for entry in refined_data2 if entry['num_gpus']==gpu_number ]
            for row in range(row_num):
                replicate_number = int( row / max(variable_lengths) )
                replicate_length = len(relevant_data[replicate_number])
                replicate_desired_entry = row % max(variable_lengths)
                if replicate_desired_entry < replicate_length:
                    data_array[row,col] = relevant_data[replicate_number][replicate_desired_entry]
                else:
                    #data_array[row,col] = np.nan
                    data_array[row,col] = np.average(relevant_data[replicate_number])
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
        
        font_size = 16
        plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
        # plot and save figures for each number of GPUs
        bin_number = 1000
        gpus = ["1GPU", "4GPU", "8GPU"]
        for gpu in gpus:
            n_col = len(self.data_df.columns) 
            if self.chosen_img_num == 1000000:
                if gpu == "1GPU":
                    xmaxes = [5,25,3,7]
                elif gpu == "4GPU":
                    xmaxes = [5,120,3,2]
                elif gpu == "8GPU":
                    xmaxes = [5,120,3,2]
            elif self.chosen_img_num == 100000:
                if gpu == "1GPU":
                    xmaxes = [14,25,3,7]
                elif gpu == "4GPU":
                    xmaxes = [5,120,3,2]
                elif gpu == "8GPU":
                    xmaxes = [5,120,3,2]
            elif self.chosen_img_num == 10000:
                if gpu == "1GPU":
                    xmaxes = [6,25,3,1.5]
                elif gpu == "4GPU":
                    xmaxes = [5,120,3,1.5]
                elif gpu == "8GPU":
                    xmaxes = [5,120,3,1.5]
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
                fig.subplots_adjust(wspace=0.3)
            for i in range(number_of_subplots):
                name = gpu + "_" + times[i]
                if number_of_subplots==4:
                    subplot_x = int("{:02b}".format(i)[0])
                    subplot_y = int("{:02b}".format(i)[1])
                    print(f"Creating plot {name} in position {subplot_x},{subplot_y}.")
                elif number_of_subplots==3:
                    print(f"Creating plot {name} in position {i}.")
                try:
                    print(f"Sum of 1000 bin histogram counts: {np.sum(np.histogram(self.data_df[[name]],bins=1000)[0])}")
                except ValueError:
                    print(f"Whoopsie. Nan error. No histogram stats reported for {name}.")
                print(f"Max value in {name}: {np.max(self.data_df[[name]])}")
                fixed_bin_width = True
                fixed_bin_number = False
                assert (fixed_bin_width and not fixed_bin_number) or (fixed_bin_number and not fixed_bin_width)
                if fixed_bin_width:
                    # compute bin cutoffs
                    data_min = np.min(self.data_df[[name]])
                    data_max = np.max(self.data_df[[name]])
                    max_increment = np.ceil(data_max-data_min)
                    bin_cutoffs = [float(data_min + increment) for increment in np.linspace( 0, max_increment, max_increment*10+1)]
                    #print(f"bin cutoffs for {name}: {bin_cutoffs}")
                    if number_of_subplots==4:
                        self.axe = self.data_df[[ name ]].plot.hist(
                            ax=self.axes[subplot_x,subplot_y],
                            bins=bin_cutoffs,
                            legend=False,
                            cmap=self.color_maps[i])
                    elif number_of_subplots==3:
                        self.axe = self.data_df[[ name ]].plot.hist(
                            ax=self.axes[i],
                            bins=bin_cutoffs,
                            legend=False,
                            cmap=self.color_maps[i])
                elif fixed_bin_number:
                    # compute bin width
                    data_min = np.min(self.data_df[[name]])
                    data_max = np.max(self.data_df[[name]])
                    bin_width = (data_max - data_min)/bin_number
                    bin_widths.append(bin_width)
                    print(f"bin_width for {name}: {bin_width}")
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
                if True:
                    # cut off tail
                    self.axe.set_xlim(xmin=0,xmax=xmaxes[i])
                    self.axe.set_xticks([0,int(xmaxes[i]/3),int(2*xmaxes[i]/3),xmaxes[i]])
                else:
                    # show full distribution
                    xmax = np.max(self.data_df[[name]])[0]
                    self.axe.set_xlim(xmin=0,xmax=xmax)
                    self.axe.set_xticks([0,int(xmax/3),int(2*xmax/3),int(xmax)])
            gpu_plot_name = gpu + "_" + self.plot_pdf_name
            plt.savefig(path.join(self.output_folder,gpu_plot_name), transparent=True)

        if False:
            import pdb; pdb.set_trace()

class BulkTimeVsGpuFigure(BaseFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, output_folder):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder)
        self.plot_title = "Many Image Semantic Segmentation Runtimes"
        if chosen_delay==5.0:
            self.plot_pdf_name = "5sdelay_image_bulk_runtimes.pdf"
        elif chosen_delay==0.5:
            self.plot_pdf_name = "0point5sdelay_image_bulk_runtimes.pdf"
        self.y_label = "Counts"

    def refine_data(self):
        # only choose relevant runs
        refined_data = []
        for entry in self.raw_data:
            if entry["start_delay"] == self.chosen_delay:
                refined_data.append(entry)
        if len(refined_data) == 0:
            raise MissingDataError
        # grab variables of interest from relevant runs
        variable_of_interest = 'time_elapsed'
        gpu_nums = [1,4,8]
        output_data = {}
        failed_img_nums = []
        for img_num in self.chosen_img_nums:
            data_lists = []
            for gpu_num in gpu_nums:
                times = [entry[variable_of_interest] for entry in refined_data if entry['num_gpus']==gpu_num and entry["num_images"]==img_num]
                data_lists.append(times)
            print(f"img_num: {img_num}, gpu_num: {gpu_num}")
            print(data_lists)
            try:
                output_data[str(img_num)] = self.format_data_for_error_plotting(data_lists)
            except ZeroDivisionError:
                # This probably means that we don't have any data for the chosen time at the chosen delay.
                print("ZeroDivisionError in format_data_for_error_plotting")
                failed_img_nums.append(img_num)
        successful_img_nums = self.chosen_img_nums
        for img_fail in failed_img_nums:
            successful_img_nums.remove(img_num)
        # create DataFrame from variables of interest
        # 2 is a magic number that derives from the format of the data returned from format_data_for_error_plotting
        col_num = len(successful_img_nums) * 2
        row_num = len(gpu_nums)
        data_array = np.zeros( (row_num, col_num) )
        for row in range(row_num):
            for col in range(col_num):
                var_num = col % len(successful_img_nums)
                var_index = int( (col - var_num) / len(successful_img_nums) )
                data_array[row,col] = output_data[ str(successful_img_nums[var_num]) ][ var_index ][ row ]
        try:
            self.data_df = pd.DataFrame(data_array, columns=["10000","100000","10000_err","100000_err"], index=["1GPU","4GPU","8GPU"])
        except ValueError:
            import pdb; pdb.set_trace()

    def plot(self, labels=None, title="multiple unstacked bar plot", **kwargs):
        """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
           labels is a list of the names of the dataframe, used for the legend title is a string for the 
           title of the plot H is the hatch used for identification of the different dataframe"""

        # preliminaries
        self.refine_data()
        self.define_colorblind_color_maps()
        
        n_col = len(self.data_df.columns) 
        fig, axes = plt.subplots(1, figsize=(20,10))
        axe = axes
        axe = self.data_df[["10000", "100000"]].plot(kind="line",
                        yerr=self.data_df[["10000_err", "100000_err"]].values.T,
                        linewidth=2,
                        stacked=False,
                        ax=axe,
                        legend=False,
                        grid=False,
                        **kwargs)  # make bar plots

        axe.set_title(title)
        axe.set_ylabel(self.y_label)
        axe.set_xlabel('Number of GPUs')
        axe.set_xticks([0,1,2])
        axe.set_xlim(-0.05,2.05)
        axe.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
        h,l = axe.get_legend_handles_labels() # get the handles we want to modify
        l1 = axe.legend(h[:n_col], l[:n_col], loc=[.425, 0.75])
        axe.add_artist(l1)
        plt.savefig(path.join(self.output_folder,self.plot_pdf_name), transparent=True)

class CostVsGpuFigure(BaseFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, pdf_label, output_folder):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder)
        self.plot_title = "Semantic Segmentation Workflow Component Runtime"
        if chosen_delay==5.0:
            self.plot_pdf_name = pdf_label + "_5sdelay_costs.pdf"
        elif chosen_delay==0.5:
            self.plot_pdf_name = pdf_label + "_0point5sdelay_costs.pdf"
        self.y_label = "Counts"

    def refine_data(self):
        # generate df
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
        variables_of_interest = ['total_node_and_networking_costs', 'cpu_node_cost', 'gpu_node_cost', 'extra_network_costs']
        gpu_nums = [1,4,8]
        output_data = {}
        for variable_of_interest in variables_of_interest:
            data_lists = []
            for gpu_num in gpu_nums:
                times = [entry[variable_of_interest] for entry in refined_data2 if entry['num_gpus']==gpu_num]
                data_lists.append(times)
            output_data[variable_of_interest] = self.format_data_for_error_plotting(data_lists)
        # create DataFrame from variables of interest
        # 2 is a magic number that derives from the format of the data returned from format_data_for_error_plotting
        col_num = len(variables_of_interest)*2
        row_num = len(gpu_nums)
        data_array = np.zeros( (row_num, col_num) )
        for row in range(row_num):
            for col in range(col_num):
                var_num = col % len(variables_of_interest)
                var_index = int( (col - var_num) / len(variables_of_interest) )
                data_array[row,col] = output_data[ variables_of_interest[var_num] ][ var_index ][ row ]
        self.data_df = pd.DataFrame(data_array, columns=["total cost", "cpu node cost","gpu node cost","network costs","total_err","cpu_err","gpu_err","network_err"], index=["1GPU","4GPU","8GPU"])

    def plot(self, labels=None, title="multiple unstacked bar plot", **kwargs):
        """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
           labels is a list of the names of the dataframe, used for the legend title is a string for the 
           title of the plot H is the hatch used for identification of the different dataframe"""

        # preliminaries
        self.refine_data()
        self.define_colorblind_color_maps()
        n_col = len(self.data_df.columns) 
        
        fig, axes = plt.subplots(1, figsize=(20,10))
        axe = axes
        axe = self.data_df[["total cost", "cpu node cost", "gpu node cost", "network costs"]].plot(kind="line",
                        yerr=self.data_df[["total_err", "cpu_err", "gpu_err", "network_err"]].values.T,
                        linewidth=2,
                        stacked=False,
                        ax=axe,
                        legend=False,
                        grid=False,
                        **kwargs)  # make bar plots

        axe.set_title(title)
        axe.set_ylabel(self.y_label)
        axe.set_xlabel('Number of GPUs')
        axe.set_xticks([0,1,2])
        axe.set_xlim(-0.05,2.05)
        axe.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
        h,l = axe.get_legend_handles_labels() # get the handles we want to modify
        l1 = axe.legend(h[:n_col], l[:n_col], loc=[.425, 0.75])
        axe.add_artist(l1)
        plt.savefig(path.join(self.output_folder,self.plot_pdf_name), transparent=True)
