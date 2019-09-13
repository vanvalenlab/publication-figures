# figures.py

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from os import path
from matplotlib.colors import LinearSegmentedColormap

from .utils import MissingDataError


class BaseFigure(object):

    def __init__(self, output_folder):
        self.font_size = 20
        self.output_folder = output_folder

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


class BaseEmpiricalFigure(BaseFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, output_folder):
        super().__init__(output_folder)
        self.raw_data = raw_data
        self.chosen_delay = chosen_delay
        self.chosen_img_num = chosen_img_num
        self.chosen_img_nums = self.chosen_img_num # alias for figures using multiple image numbers

    # TODO: this is for symmetric data, but our data is necessarily non-negative
    def format_data_for_error_plotting(self, data_lists):
        means = []
        std_devs = []
        std_errs_of_means = []
        percentiles_975 = []
        percentiles_025 = []
        one_sided_95_percent_intervals = []
        for datum_list in data_lists:
            if len(datum_list) > 1:
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
                std_devs.append(std_dev)
                std_err_of_mean = std_dev/(len(datum_list)**(1/2))
                std_errs_of_means.append(std_err_of_mean)

                # compute 95% confidence intervals
                percentile_975 = mean + (std_err_of_mean * 1.96)
                percentile_025 = mean - (std_err_of_mean * 1.96)
                percentiles_975.append(percentile_975)
                percentiles_025.append(percentile_025)
                one_sided_95_percent_intervals.append(std_err_of_mean * 1.96)
            elif len(datum_list)==1:
                # TODO: this is a hack to deal with the 1,000,000 image situations, where we don't have replicates
                mean = np.nanmean(datum_list)
                means.append(mean)
                one_sided_95_percent_intervals.append(0)
                std_devs.append(0)
            else:
                # TODO: this is another hack to deal with the fact that we don't even have complete series of data for the 1,000,000 image case
                # len(datum_list)==0 ?
                # raise error for now
                #raise ValueError("No data???")
                mean = np.nan
                means.append(mean)
                one_sided_95_percent_intervals.append(mean)
                std_devs.append(mean)
        #return (means, one_sided_95_percent_intervals)
        return (means, std_devs)


class ImageTimeVsGpuFigure(BaseEmpiricalFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, pdf_label, output_folder):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder)
        self.plot_title = "Semantic Segmentation Workflow Component Runtime"
        if self.chosen_delay==5.0:
            self.plot_pdf_name = pdf_label + "_5sdelay_image_runtimes.pdf"
        elif self.chosen_delay==0.5:
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
        if self.chosen_img_num==1000000:
            replicates = 1
        else:
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
            # for the 1,000,000 image cases where we don't have any data
            if len(relevant_data)==0:
                data_array[:,col] = np.nan
            else:
                for row in range(row_num):
                    replicate_number = int( row / max(variable_lengths) )
                    try:
                        replicate_length = len(relevant_data[replicate_number])
                    except IndexError:
                        import pdb; pdb.set_trace()
                    replicate_desired_entry = row % max(variable_lengths)
                    if replicate_desired_entry < replicate_length:
                        try:
                            data_array[row,col] = relevant_data[replicate_number][replicate_desired_entry]
                        except ValueError:
                            for i, datum in enumerate(relevant_data[replicate_number]):
                                if isinstance(datum, str):
                                    relevant_data[replicate_number][i] = np.nan
                            relevant_data_mean = np.nanmean(relevant_data[replicate_number])
                            for i, datum in enumerate(relevant_data[replicate_number]):
                                if np.isnan(datum):
                                    relevant_data[replicate_number][i] = relevant_data_mean
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
        
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels
        # plot and save figures for each number of GPUs
        bin_number = 1000
        gpus = ["1GPU", "4GPU", "8GPU"]
        for gpu in gpus:
            # choose type of time measurement for each subplot
            times = ["network", "prediction", "postprocess"]
            number_of_subplots = len(times)
            # yet again, accommodating partial 1,000,000 image data series
            no_plot = False
            for i in range(number_of_subplots):
                name = gpu + "_" + times[i]
                try:
                    if self.data_df[[ name ]].isnull().all()[0]:
                        pass
                    else:
                        break
                except ValueError:
                    import pdb; pdb.set_trace()
                if i==max(range(number_of_subplots)):
                    # we don't need to be plotting this becasue we have no data
                    no_plot = True
            if no_plot:
                continue
            n_col = len(self.data_df.columns) 
            if self.chosen_img_num == 1000000:
                if gpu == "1GPU":
                    xmaxes = [5,25,3,7]
                    custom_xticks0 = [0,1,2,3,4,5]
                elif gpu == "4GPU":
                    xmaxes = [5,120,3,2]
                    custom_xticks0 = [0,1,2,3,4,5]
                elif gpu == "8GPU":
                    xmaxes = [5,120,3,2]
                    custom_xticks0 = [0,1,2,3,4,5]
            elif self.chosen_img_num == 100000:
                if gpu == "1GPU":
                    xmaxes = [14,25,3,7]
                elif gpu == "4GPU":
                    xmaxes = [5,120,3,2]
                    custom_xticks0 = [0,1,2,3,4,5]
                elif gpu == "8GPU":
                    xmaxes = [5,120,3,2]
                    custom_xticks0 = [0,1,2,3,4,5]
            elif self.chosen_img_num == 10000:
                if gpu == "1GPU":
                    xmaxes = [6,25,3,1.5]
                elif gpu == "4GPU":
                    xmaxes = [5,120,3,1.5]
                    custom_xticks0 = [0,1,2,3,4,5]
                elif gpu == "8GPU":
                    xmaxes = [5,120,3,1.5]
                    custom_xticks0 = [0,1,2,3,4,5]

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
                if i==0:
                    self.axe.set_ylabel(self.y_label)
                else:
                    self.axe.set_ylabel("")
                if i==1:
                    self.axe.set_xlabel('Time (s)')
                else:
                    self.axe.set_xlabel("")
                self.axe.set_ylim(ymin=0)
                if True:
                    # cut off tail
                    self.axe.set_xlim(xmin=0,xmax=xmaxes[i])
                    if i==0:
                        try:
                            custom_xticks0
                        except NameError:
                            self.axe.set_xticks([0,int(xmaxes[i]/3),int(2*xmaxes[i]/3),xmaxes[i]])
                        else:
                            self.axe.set_xticks(custom_xticks0)
                    else:
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


class BulkTimeVsGpuFigure(BaseEmpiricalFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, output_folder):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder)
        self.plot_title = "Many Image Semantic Segmentation Runtimes"
        if self.chosen_delay==5.0:
            self.plot_pdf_name = "5sdelay_image_bulk_runtimes.pdf"
        elif self.chosen_delay==0.5:
            self.plot_pdf_name = "0point5sdelay_image_bulk_runtimes.pdf"
        self.y_label = "Time (min)"

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
            except ValueError:
                import pdb; pdb.set_trace()
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
            if self.chosen_delay==5.0:
                data_array = np.delete(data_array,5,1)
                data_array = np.delete(data_array,2,1)
                self.data_df = pd.DataFrame(data_array, columns=["10000","100000","10000_err","100000_err"], index=["1GPU","4GPU","8GPU"])
            elif self.chosen_delay==0.5:
                self.data_df = pd.DataFrame(data_array, columns=["10000","100000","1000000","10000_err","100000_err","1000000_err"], index=["1GPU","4GPU","8GPU"])
        except ValueError:
            import pdb; pdb.set_trace()

    def plot(self, labels=None, **kwargs):
        """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
           labels is a list of the names of the dataframe, used for the legend title is a string for the 
           title of the plot H is the hatch used for identification of the different dataframe"""

        # preliminaries
        self.refine_data()
        self.define_colorblind_color_maps()
        
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels
        
        # plot data
        xmin=0.4
        xmax=1.6
        bar_width = 0.15
        x_ticks = [0.5,1,1.5]
        if self.chosen_delay==5.0:
            x_ticks1 = [x_tick-bar_width/2 for x_tick in x_ticks]
            x_ticks2 = [x_tick+bar_width/2 for x_tick in x_ticks]
        elif self.chosen_delay==0.5:
            x_ticks1 = [x_tick-bar_width for x_tick in x_ticks]
            x_ticks2 = [x_tick for x_tick in x_ticks]
            x_ticks3 = [x_tick+bar_width for x_tick in x_ticks]
        n_col = len(self.data_df.columns) 
        fig, axes = plt.subplots(1, figsize=(10,15))
        axe = axes
        if False:
            axe = self.data_df[["10000", "100000"]].plot(kind="bar",
                            yerr=self.data_df[["10000_err", "100000_err"]].values.T,
                            linewidth=2,
                            stacked=False,
                            ax=axe,
                            legend=False,
                            grid=False,
                            width=bar_width,
                            **kwargs)  # make bar plots
        axe.errorbar(x_ticks,
            self.data_df.loc[["1GPU","4GPU","8GPU"],"10000"],
            yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"10000_err"],
            linestyle="None",
            fmt='o',
            color=self.color_maps[0](0))
        axe.errorbar(x_ticks,
            self.data_df.loc[["1GPU","4GPU","8GPU"],"100000"],
            yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"100000_err"],
            linestyle="None",
            fmt='o',
            color=self.color_maps[0](1))
        if self.chosen_delay==0.5:
            axe.errorbar(x_ticks,
                self.data_df.loc[["1GPU","4GPU","8GPU"],"1000000"],
                yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"1000000_err"],
                linestyle="None",
                fmt='o',
                color=self.color_maps[0](2))

        print(f"Bulk runtimes for {self.chosen_delay}:")
        print(f'10,000 images: {self.data_df.loc[["1GPU","4GPU","8GPU"],"10000"]}')
        print(f'100,000 images: {self.data_df.loc[["1GPU","4GPU","8GPU"],"100000"]}')
        try:
            print(f'1,000,000 images: {self.data_df.loc[["1GPU","4GPU","8GPU"],"1000000"]}')
        except KeyError:
            pass
        axe.set_title(self.plot_title)
        axe.set_ylabel(self.y_label)
        axe.set_xlabel('Number of GPUs')
        axe.set_xticks(x_ticks)
        axe.set_xlim(xmin,xmax)
        axe.set_ylim(ymin=10,ymax=1000)
        axe.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
        axe.set_yscale("log")
        #h,l = axe.get_legend_handles_labels() # get the handles we want to modify
        #l1 = axe.legend(h[:n_col], l[:n_col], loc=[.425, 0.75])
        #axe.add_artist(l1)
        if self.chosen_delay==5.0:
            axe.legend(["10,000 Images", "100,000 Images"], prop={'size':self.font_size})
        elif self.chosen_delay==0.5:
            axe.legend(["10,000 Images", "100,000 Images", "1,000,000 Images"], prop={'size':self.font_size})
        plt.savefig(path.join(self.output_folder,self.plot_pdf_name), transparent=True)


class CostVsGpuFigure(BaseEmpiricalFigure):

    def __init__(self, raw_data, chosen_delay, chosen_img_num, pdf_label, output_folder):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder)
        self.plot_title = "Semantic Segmentation Costs"
        if self.chosen_delay==5.0:
            self.plot_pdf_name = pdf_label + "_5sdelay_costs.pdf"
        elif self.chosen_delay==0.5:
            self.plot_pdf_name = pdf_label + "_0point5sdelay_costs.pdf"
        self.y_label = "Cost (USD)"

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
        
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels
        
        fig, axes = plt.subplots(1, figsize=(15,10))
        axe = axes
        if False:
            axe = self.data_df[["total cost", "cpu node cost", "gpu node cost", "network costs"]].plot(kind="bar",
                        yerr=self.data_df[["total_err", "cpu_err", "gpu_err", "network_err"]].values.T,
                        linewidth=2,
                        stacked=False,
                        ax=axe,
                        legend=False,
                        grid=False,
                        **kwargs)  # make bar plots
        xmin=-0.25
        xmax=2.25
        bar_width = 0.35
        x_ticks = [0.18,1,1.82]
        x_ticks1 = [x_tick-(bar_width*1.5) for x_tick in x_ticks]
        x_ticks2 = [x_tick-(bar_width*0.5) for x_tick in x_ticks]
        x_ticks3 = [x_tick+(bar_width*0.5) for x_tick in x_ticks]
        x_ticks4 = [x_tick+(bar_width*1.5) for x_tick in x_ticks]
        n_col = len(self.data_df.columns) 
        if False:
            axe.bar(
                x_ticks1,
                self.data_df.loc[["1GPU","4GPU","8GPU"],"network costs"],
                yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"network_err"],
                width=bar_width,
                color=self.color_maps[0](0),
                bottom=None)
            axe.bar(
                x_ticks2,
                self.data_df.loc[["1GPU","4GPU","8GPU"],"gpu node cost"],
                yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"gpu_err"],
                width=bar_width,
                color=self.color_maps[0](1),
                bottom=None)
            axe.bar(
                x_ticks3,
                self.data_df.loc[["1GPU","4GPU","8GPU"],"cpu node cost"],
                yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"cpu_err"],
                width=bar_width,
                color=self.color_maps[0](2),
                bottom=None)
            axe.bar(
                x_ticks4,
                self.data_df.loc[["1GPU","4GPU","8GPU"],"total cost"],
                yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"total_err"],
                width=bar_width,
                color=self.color_maps[0](3),
                bottom=None)
        #import pdb; pdb.set_trace()
        axe.bar(
            x_ticks,
            self.data_df.loc[["1GPU","4GPU","8GPU"],"network costs"],
            yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"network_err"],
            width=bar_width,
            color=self.color_maps[0](0),
            bottom=None)
        axe.bar(
            x_ticks,
            self.data_df.loc[["1GPU","4GPU","8GPU"],"gpu node cost"],
            yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"gpu_err"],
            width=bar_width,
            color=self.color_maps[0](1),
            bottom=self.data_df.loc[["1GPU","4GPU","8GPU"],"network costs"])
        axe.bar(
            x_ticks,
            self.data_df.loc[["1GPU","4GPU","8GPU"],"cpu node cost"],
            yerr=self.data_df.loc[["1GPU","4GPU","8GPU"],"cpu_err"],
            width=bar_width,
            color=self.color_maps[0](2),
            bottom=self.data_df.loc[["1GPU","4GPU","8GPU"],"gpu node cost"]+self.data_df.loc[["1GPU","4GPU","8GPU"],"network costs"])


        axe.set_title(self.plot_title)
        axe.set_ylabel(self.y_label)
        axe.set_xlabel('Number of GPUs')
        axe.set_xticks(x_ticks)
        axe.set_xlim(xmin,xmax)
        axe.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
        #h,l = axe.get_legend_handles_labels() # get the handles we want to modify
        #l1 = axe.legend(h[:n_col], l[:n_col], loc=[.425, 0.75])
        #axe.add_artist(l1)
        axe.legend(["Network and Data Costs", "GPU Node Cost", "CPU Node Cost"],
            prop={'size':self.font_size},
            loc=[0.30,0.75])
        plt.savefig(path.join(self.output_folder,self.plot_pdf_name), transparent=True)


class OptimalGpuNumberFigure(BaseFigure):

    def __init__(self, output_folder):
        super().__init__(output_folder)
        self.plot_pdf_name = "optimal_gpu_nums.pdf"
        self.define_colorblind_color_maps()

    def plot(self, labels=None, **kwargs):
        """This function plots optimal contours (for a given number of GPUs) on a
        (network data transfer speed)[Mb/s] vs. (tensorflow image processing speed)[images/s] plot."""

        # matplotlib comfiguration options
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels

        # declare data for plotting
        image_size_bits = 8000000 # (bits/image), 1000 pixels x 1000 pixels x 8 bits
        bits_to_megabits_conversion_factor = 1024*1024 # bits/Mb
        #prediction_speed = [ 0.1, 0.25, 0.5, 1, 2, 5, 10, 100 ] # images/sec
        #prediction_speed = np.logspace(-1,2,num=200) # images/sec
        prediction_speed = np.linspace(0.1,100,num=1000) # images/sec
        #upload_speed = np.logspace(1,3.5,num=50) # [10,~3000], Mb/s 
        upload_speed =  [ 10, 50, 250, 700, 1200, 1800, 2400, 3000 ] # Mb/s 
        #gpu_nums = [ 1, 2, 4, 8, 16 ]
        all_data = np.zeros( (len(prediction_speed), len(upload_speed)) )
        for i, rate in enumerate(upload_speed):
            data = [ round( (rate/x)*(image_size_bits/bits_to_megabits_conversion_factor) - 0.5 ) for x in prediction_speed ]
            all_data[:,i] = data        
        
        # plot parameters
        title = "Optimal GPU Numbers for Given Network Speeds and Tensorflow Processing Speeds"
        xlabel = "Prediction Speed (images/s)"
        ylabel = "Number of GPUs"
        xmin=0
        xmax=max(prediction_speed)+1
        ymin=0.98
        ymax=50
        #ymax=np.max(all_data) + 1
        #x_ticks = [0.5,1,1.5]
        
        # create and configure plot
        fig, axis = plt.subplots(1, figsize=(20,10))
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xlim(xmin,xmax)
        axis.set_ylim(ymin,ymax)
        #axis.set_xticks(x_ticks)
        #axis.set_xscale("log")

        # plot data
        for i, rate in enumerate(upload_speed):
            if rate==1:
                plot_label = str(rate) + " Mb/s"
            else:
                plot_label = str(rate) + " Mb/s"
            color_index = i % self.color_maps[0].N
            #import pdb; pdb.set_trace()
            axis.plot(
                prediction_speed,
                all_data[:,i],
                color=self.color_maps[0](color_index),
                label=plot_label)
        # add legend
        legend_labels = []
        for rate in upload_speed:
            if rate==1:
                label = str(rate) + " Mb/s"
            else:
                label = str(rate) + " Mb/s"
            legend_labels.append(label)
        axis.legend(legend_labels, prop={'size':self.font_size})
        plt.savefig(path.join(self.output_folder,self.plot_pdf_name), transparent=True)
