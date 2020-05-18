# figures.py
""" The terminal classes in this file are each intended to create a different type of figure.
    They all inherit from either `BaseEmpiricalFigure` (for figures relying on benchmarking data)
    or `BaseFigure` (for figures explaining some theoretical principle). It's worth noting that the
    `BaseEmpiricalFigure` class inherits from the `BaseFigure` class.

"""


import logging
from os import path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .utils import MissingDataError


class BaseFigure(object):
    """ This is the base class for all figures.

    Args:
        output_folder (str, optional): The folder into which the figures should be written
        logger_name (str, optional): the name of the class whose logger is being created

    Attributes:
        font_size (int): size of font to pass to matplotlib for plotting
        output_folder (str): path, local or absolute, of folder to place figures in; must already
            exist
        color_maps ([matplotlib.colors.LinearSegmentedColorMap]): a list of color maps, each
            containing a uniquely-ordered list of all the color in the same blind-friendly color
            pallette
        logger (logging.getLogger()): logger retrieved using the logger_name arg

    """
    def __init__(self, output_folder="outputs", logger_name="BaseFigure"):
        self.font_size = 20
        self.output_folder = output_folder
        self.color_maps = self.define_colorblind_color_maps()
        self.logger = self.configure_logging(logger_name)

    @staticmethod
    def define_colorblind_color_maps():
        """ Create a list of color blind-friendly color maps.

        This method creates a series of colormaps for use in making plots intelligible by
        people who are red-green color blind. This color map originally comes from
        Bang Wong, Nature Methods, volume 8, page 441 (2011)

        Returns:
            [matplotlib.colors.LinearSegmentedColorMap]: a list of color maps, each containing
                a uniquely-ordered list of all the color in the same blind-friendly color
                pallette

        """
        colors = [(86/255, 180/255, 233/255),
                  (0, 158/255, 115/255),
                  (204/255, 121/255, 167/255),
                  (213/255, 94/255, 0),
                  (230/255, 159/255, 0),
                  (0, 114/255, 178/255),
                  (213/255, 94/255, 0),
                  (0, 0, 0)]
        n_bins = len(colors)
        color_maps = []
        for cut_point in range(n_bins):
            cmap_name = 'colorblind' + str(cut_point+1)
            unflattened_new_color_list = [colors[cut_point:] + colors[:cut_point]]
            new_color_list = [color
                              for color_list in unflattened_new_color_list
                              for color in color_list]
            cmap = LinearSegmentedColormap.from_list(cmap_name, new_color_list, N=n_bins)
            color_maps.append(cmap)
        return color_maps

    @staticmethod
    def configure_logging(logger_name):
        """ This function configures logging for the whole instance.

        Args:
            logger_name (str): the name of the class whose logger is being created

        Returns:
            logging.getLogger(): logger retrieved using the given logger_name

        """
        class_logger = logging.getLogger(logger_name)
        class_logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh.setFormatter(formatter)
        class_logger.addHandler(sh)
        class_logger.propagate = False
        fh = logging.FileHandler("figure_creation.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        class_logger.addHandler(fh)
        return class_logger


class BaseEmpiricalFigure(BaseFigure):
    """ This is the base figure class for all figures that rely on experimental data.

    Args:
        raw_data ([{}]): list of dictionaries, each one representing a different
            benchmarking run
        chosen_delay (float): the delay used when uploading file for the desired runs.
            NB: The values used indicate how many seconds our simulated uploading pipeline
            waited between simulating 100 image zip file uploads. Given that our zip files were
            ~150Mb each, the delays of 0.5s and 5.0s correspond to ~240Mbps and 2,400Mbps.
        chosen_img_num (int): the number of images processed in the desired runs
        output_folder (str): the folder into which the figures should be written
        logger_name (str, optional): the name of the class whose logger is being created

    Attributes:
        raw_data ([{}]): list of dictionaries, each one representing a different benchmarking run
        chosen_delay (float): the delay used when uploading file for the desired runs.
            NB: The values used indicate how many seconds our simulated uploading pipeline
            waited between simulating 100 image zip file uploads. Given that our zip files were
            ~150Mb each, the delays of 0.5s and 5.0s correspond to ~240Mbps and 2,400Mbps.
        chosen_img_num (int or [int]): the number of images processed in the desired runs
        chosen_img_nums (int or [int]): alias for figures using mutliple image numbers

    """
    def __init__(self, raw_data, chosen_delay, chosen_img_num, output_folder,
                 logger_name="BaseEmpiricalFigure"):
        super().__init__(output_folder, logger_name=logger_name)
        self.raw_data = raw_data
        self.chosen_delay = chosen_delay
        self.chosen_img_num = chosen_img_num
        self.chosen_img_nums = self.chosen_img_num # alias for figures using multiple image numbers

    @staticmethod
    def format_data_for_error_plotting(data_lists):
        """ Compute error bars for plots.

        This method is used to compute error bars for plots. It's only relevant for run
        conditions that we have triplicate data for. Since it's also called for singleton
        run conditions, we have some hacks in the code to get it to execute on them
        without throwing an exception.

        Args:
            data_list ([{}]): list of dictionaries, one dictionary for each replicate of the chosen
                run conditions

        Todo:
            * this is for symmetric data, but our data is necessarily non-negative

        """
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
                std_dev = ((1/(len(datum_list)-1)) * squared_deviations)**(1/2)
                std_devs.append(std_dev)
                std_err_of_mean = std_dev/(len(datum_list)**(1/2))
                std_errs_of_means.append(std_err_of_mean)

                # compute 95% confidence intervals
                percentile_975 = mean + (std_err_of_mean * 1.96)
                percentile_025 = mean - (std_err_of_mean * 1.96)
                percentiles_975.append(percentile_975)
                percentiles_025.append(percentile_025)
                one_sided_95_percent_intervals.append(std_err_of_mean * 1.96)
            # this is a hack to deal with the 1,000,000 image situations,
            # where we don't have replicates
            elif len(datum_list) == 1:
                mean = np.nanmean(datum_list)
                means.append(mean)
                one_sided_95_percent_intervals.append(0)
                std_devs.append(0)
            # this is another hack to deal with the fact that we don't have
            # a complete series of data for the 1,000,000 image case
            else:
                mean = np.nan
                means.append(mean)
                one_sided_95_percent_intervals.append(mean)
                std_devs.append(mean)
        #return (means, one_sided_95_percent_intervals)
        return (means, std_devs)


class ImageTimeVsGpuFigure(BaseEmpiricalFigure):
    """ Output one three-panel image time vs. #GPU figure for each GPU number.

    This class produces a three-panel figure, for each GPU number in self.gpus, showing the
    image-level distributions 1) Data Transfer Times, 2) Tensorflow Processing Times, and
    3) Postprocessing Times for the aggregated data of all runs with a given set of
    (delay)x(image number) conditions.

    Args:
        raw_data ([{}]): list of dictionaries, each one representing a different
            benchmarking run
        chosen_delay (float): the delay used when uploading file for the desired runs. NB: The
            values used indicate how many seconds our simulated uploading pipeline waited
            between simulating 100 image zip file uploads. Given that our zip files were ~150Mb
            each, the delays of 0.5s and 5.0s correspond to ~240Mbps and 2,400Mbps.
        chosen_img_num (int): the number of images processed in the desired runs
        pdf_label (str): base of title for the saved figure's PDF file
        output_folder (str): the folder into which the figures should be written
        bins_fixed_width_or_number (str, optional): if set to "number", fix the number of bins in a
            histogram, while letting the widths of the bins vary; if set to "width", fix the
            widths of the bins, while letting the number vary
        cut_off_tails (bool, optional): whether or not to cut off most of the length of the long
            tails of the histograms at a predetermined point
        logger_name (str, optional): the name of the class whose logger is being created

    Attributes:
        plot_title (str): title, to be used on the top of each saved matplotlib figure
        plot_pdf_name (str): base filename, used as the output filename after being predended with
            the appropriate number of GPUs
        y_label (str): y-axis label for figures; currently set to "Counts" and not
            user-configurable
        gpus ([str]): Strings corresponding to the numbers of GPUs used in the runs of current
            interest. The exact form of the strings reflects the convention used in the benchmarking
            filenames and, through this class, is propogated to the output file names. Currently set
            to ["1GPU", "4GPU", "8GPU"] and not user-configurable.
        fixed_bin_property (str): if set to "number", fix the number of bins in a
            histogram, while letting the widths of the bins vary; if set to "width", fix the
            widths of the bins, while letting the number vary
        cut_off_tails (bool): whether or not to cut off most of the length of the long tails of the
            histograms at a predetermined point
        data_df (None, or pandas.DataFrame): initialized to None, but eventually used to store all
            cleaned and properly formatted data for the figures

    Outputs:
        [Sick PDF plot.]

    """
    def __init__(self, raw_data, chosen_delay, chosen_img_num, pdf_label, output_folder,
                 bins_fixed_width_or_number="width", cut_off_tails=True,
                 logger_name="ImageTimeVsGpuFigure"):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder,
                         logger_name=logger_name)
        self.plot_title = "Semantic Segmentation Workflow Component Runtime"
        if self.chosen_delay == 5.0:
            self.plot_pdf_name = pdf_label + "_5sdelay_image_runtimes.pdf"
        elif self.chosen_delay == 0.5:
            self.plot_pdf_name = pdf_label + "_0point5sdelay_image_runtimes.pdf"
        self.y_label = "Counts"
        self.gpus = ["1GPU", "4GPU", "8GPU"]
        assert bins_fixed_width_or_number in ("width", "number")
        self.fixed_bin_property = bins_fixed_width_or_number
        assert cut_off_tails in (True, False)
        self.cut_off_tails = cut_off_tails
        self.data_df = None

    def do_we_have_data(self, number_of_subplots, gpu, times):
        """ Check whether we have any data for these run conditions.

        This function looks at self.data_df[[name]] to see whether it contains any data.
        If not, we return False, which instructs self.plot not to create a plot for the chosen
        conditions. This is important when dealing with incomplete data series, such as our
        1,000,000 image runs at both 0.5s and 5.0s delay.

        Returns:
            (bool): "True" if we have data; "False" if not

        """
        for i in range(number_of_subplots):
            name = gpu + "_" + times[i]
            if self.data_df[[name]].isnull().all()[0]:
                continue
            # if we make it here, we at least have some data, so let's plot
            return True
        # we don't need to be plotting this because we have no data
        return False

    def refine_data(self):
        """ Keep only data from desired runs.

        Starting with all data from all runs, pare data down to runs from the appropriate
        (delay)x(image number) conditions.

        Raises:
            MissingDataError: If we don't have any data for a chosen combination of upload
                delay and image number conditions, then we raise this exception.

        Todo:
            * simplify series padding, which occurs within `for col in range(col_num)` loop
                here and in one block in graph_creation.py

        """
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
            self.logger.error(f"MissingDataError raised in ImageViewVsGpuFigure for " + \
                          f"delay {self.chosen_delay} and image number {self.chosen_img_num}.")
            raise MissingDataError

        # determine desired shape of output DataFrame
        variables_of_interest = ['all_total_times',
                                 'all_network_times',
                                 'all_upload_times',
                                 'all_prediction_times',
                                 'all_postprocess_times',
                                 'all_download_times']
        gpu_nums = [1, 4, 8]
        if self.chosen_img_num == 1000000:
            replicates = 1
        else:
            replicates = 3
        variable_lengths = []
        for variable_of_interest in variables_of_interest:
            for gpu_num in gpu_nums:
                run_times = [entry[variable_of_interest]
                             for entry in refined_data2 if entry['num_gpus'] == gpu_num]
                for run in run_times:
                    variable_lengths.append(len(run))
        row_num = max(variable_lengths)*replicates
        col_num = len(gpu_nums)*len(variables_of_interest)

        # create DataFrame of predetermined size from variables of interest in relevant runs
        data_array = np.zeros((row_num, col_num))
        for col in range(col_num):
            variable_name = variables_of_interest[col % len(variables_of_interest)]
            gpu_number = gpu_nums[int(col/len(variables_of_interest)) % len(gpu_nums)]
            relevant_data = [entry[variable_name]
                             for entry in refined_data2 if entry['num_gpus'] == gpu_number]
            # for the 1,000,000 image cases where we don't have any data
            if len(relevant_data) == 0:
                data_array[:, col] = np.nan
            else:
                for row in range(row_num):
                    replicate_number = int(row / max(variable_lengths))
                    replicate_length = len(relevant_data[replicate_number])
                    replicate_desired_entry = row % max(variable_lengths)
                    if replicate_desired_entry < replicate_length:
                        try:
                            data_array[row, col] = \
                                relevant_data[replicate_number][replicate_desired_entry]
                        except ValueError:
                            for i, datum in enumerate(relevant_data[replicate_number]):
                                if isinstance(datum, str):
                                    relevant_data[replicate_number][i] = np.nan
                            relevant_data_mean = np.nanmean(relevant_data[replicate_number])
                            for i, datum in enumerate(relevant_data[replicate_number]):
                                if np.isnan(datum):
                                    relevant_data[replicate_number][i] = relevant_data_mean
                            data_array[row, col] = \
                                relevant_data[replicate_number][replicate_desired_entry]
                    else:
                        data_array[row, col] = np.average(relevant_data[replicate_number])
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
        """ Plot desired figure(s).

        This method first calls self.refine_data() for pare the raw data down to only the runs
        of current interest. Then, for each chosen GPU number, it uses matplotlib to make a
        three-panel plot of image-level histograms of data transfer times, Tensorflow response
        times, and postprocessing times.

        Outputs:
            [PDF figure(s).]

        """
        # pare data from all runs down to just the runs of current interest
        self.refine_data()

        # configure plots
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels
        bin_number = 1000
        times = ["network", "prediction", "postprocess"]
        number_of_subplots = len(times)

        # plot and save figures for each number of GPUs
        for gpu in self.gpus:
            # yet again, accommodating our 1,000,000 image partial data series
            # if we don't have data for these conditions, then self.do_we_have_data returns True
            if not self.do_we_have_data(number_of_subplots, gpu, times):
                continue
            # if we have data, choose x axis extent and, if necessary, xticks
            if self.chosen_img_num == 1000000:
                if gpu == "1GPU":
                    xmaxes = [5, 25, 3, 7]
                    custom_xticks = [0, 1, 2, 3, 4, 5]
                elif gpu == "4GPU":
                    xmaxes = [5, 120, 3, 2]
                    custom_xticks = [0, 1, 2, 3, 4, 5]
                elif gpu == "8GPU":
                    xmaxes = [5, 120, 3, 2]
                    custom_xticks = [0, 1, 2, 3, 4, 5]
            elif self.chosen_img_num == 100000:
                if gpu == "1GPU":
                    xmaxes = [14, 25, 3, 7]
                elif gpu == "4GPU":
                    xmaxes = [5, 120, 3, 2]
                    custom_xticks = [0, 1, 2, 3, 4, 5]
                elif gpu == "8GPU":
                    xmaxes = [5, 120, 3, 2]
                    custom_xticks = [0, 1, 2, 3, 4, 5]
            elif self.chosen_img_num == 10000:
                if gpu == "1GPU":
                    xmaxes = [6, 25, 3, 1.5]
                elif gpu == "4GPU":
                    xmaxes = [5, 120, 3, 1.5]
                    custom_xticks = [0, 1, 2, 3, 4, 5]
                elif gpu == "8GPU":
                    xmaxes = [5, 120, 3, 1.5]
                    custom_xticks = [0, 1, 2, 3, 4, 5]

            bin_widths = []
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
            fig.subplots_adjust(wspace=0.3)
            # individual plot construction
            for i in range(number_of_subplots):
                name = gpu + "_" + times[i]
                self.logger.debug(f"Creating plot {name} in position {i}.")
                self.logger.debug(f"Sum of 1000 bin histogram counts: " +
                                  f"{np.sum(np.histogram(self.data_df[[name]],bins=1000)[0])}")
                self.logger.debug(f"Max value in {name}: {np.max(self.data_df[[name]])[0]}")
                # create plots in memory
                if self.fixed_bin_property == "width":
                    # compute bin cutoffs
                    data_min = np.min(self.data_df[[name]])
                    data_max = np.max(self.data_df[[name]])
                    max_increment = np.int(np.ceil(data_max-data_min))
                    bin_cutoffs = [float(data_min + increment)
                                   for increment in
                                   np.linspace(0, max_increment, max_increment * 10 + 1)]
                    axis = self.data_df[[name]].plot.hist(
                        ax=axes[i],
                        bins=bin_cutoffs,
                        legend=False,
                        cmap=self.color_maps[i])
                elif self.fixed_bin_property == "number":
                    # compute bin width
                    data_min = np.min(self.data_df[[name]])
                    data_max = np.max(self.data_df[[name]])
                    bin_width = (data_max - data_min)/bin_number
                    bin_widths.append(bin_width)
                    self.logger.debug(f"bin_width for {name}: {bin_width}")
                    axis = self.data_df[[name]].plot.hist(
                        ax=axes[i],
                        bins=bin_number,
                        legend=False,
                        cmap=self.color_maps[i])
                # configure plot aesthetics
                if "postprocess" in times[i]:
                    plot_subtitle = "Postprcessing Runtime"
                elif "prediction" in times[i]:
                    plot_subtitle = "Tensorflow Serving Response Time"
                elif "network" in times[i]:
                    plot_subtitle = "Data Transfer Time"
                axis.set_title(plot_subtitle)
                if i == 0:
                    axis.set_ylabel(self.y_label)
                else:
                    axis.set_ylabel("")
                if i == 1:
                    axis.set_xlabel('Time (s)')
                else:
                    axis.set_xlabel("")
                axis.set_ylim(ymin=0)
                if self.cut_off_tails:
                    # cut off tail
                    axis.set_xlim(xmin=0, xmax=xmaxes[i])
                    if i == 0:
                        try:
                            custom_xticks
                        except NameError:
                            axis.set_xticks([0,
                                             int(xmaxes[i]/3),
                                             int(2*xmaxes[i]/3),
                                             xmaxes[i]])
                        else:
                            axis.set_xticks(custom_xticks)
                    else:
                        axis.set_xticks([0,
                                         int(xmaxes[i]/3),
                                         int(2*xmaxes[i]/3),
                                         xmaxes[i]])
                else:
                    # show full distribution
                    xmax = np.max(self.data_df[[name]])[0]
                    axis.set_xlim(xmin=0, xmax=xmax)
                    axis.set_xticks([0,
                                     int(xmax/3),
                                     int(2*xmax/3),
                                     int(xmax)])
                self.logger.debug("")
            gpu_plot_name = gpu + "_" + self.plot_pdf_name
            plt.savefig(path.join(self.output_folder, gpu_plot_name), transparent=True)
            plt.close()

        # brag
        self.logger.info(f"Created image times histograms for delay {self.chosen_delay}" +
                         f"image number {self.chosen_img_num} GPU {gpu}.")


class BulkTimeVsGpuFigure(BaseEmpiricalFigure):
    """ Create a scatterplot of runtime vs. #GPU.

    This class creates a single chart of runtimes for all runs with a given upload delay time.
    The data is plotted as a scatterplot with multiple series, one series for the averaged
    times for all runs of with a given number of images, with the number of GPUs used in the
    runs being the x-axis.

    Args:
        raw_data ([{}]): list of dictionaries, each one representing a different
            benchmarking run
        chosen_delay (float): the delay used when uploading file for the desired runs. N.B.:
            The values used indicate how many seconds our simulated uploading pipeline waited
            between simulating 100 image zip file uploads. Given that our zip files were ~150Mb
            each, the delays of 0.5s and 5.0s correspond to ~240Mbps and 2,400Mbps.
        chosen_img_num (int): the number of images processed in the desired runs
        output_folder (str): the folder into which the figures should be written
        logger_name (str, optional): the name of the class whose logger is being created

    Attributes:
        plot_title (str): title, to be used on the top of the saved matplotlib figure
        plot_pdf_name (str): output filename
        y_label (str): y-axis label for the figure; currently set to "Time (min)" and not
            user-configurable
        data_df (None, or pandas.DataFrame): initialized to None, but eventually used to store all
            cleaned and properly formatted data for the figures

    Outputs:
        [Sick PDF plot.]

    """
    def __init__(self, raw_data, chosen_delay, chosen_img_num, output_folder,
                 logger_name="BulktimeVsGpuFigure"):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder,
                         logger_name=logger_name)
        self.plot_title = "Many Image Semantic Segmentation Runtimes"
        if self.chosen_delay == 5.0:
            self.plot_pdf_name = "5sdelay_image_bulk_runtimes.pdf"
        elif self.chosen_delay == 0.5:
            self.plot_pdf_name = "0point5sdelay_image_bulk_runtimes.pdf"
        self.y_label = "Time (min)"
        self.data_df = None

    def refine_data(self):
        """ Keep only data from desired runs.

        Starting with all data from all runs, pare data down to runs from the appropriate
        (delay) conditions.

        Raises:
            MissingDataError: If we don't have any data for a chosen combination of upload
                delay and image number conditions, then we raise this exception.

        """
        # only choose relevant runs
        refined_data = []
        for entry in self.raw_data:
            if entry["start_delay"] == self.chosen_delay:
                refined_data.append(entry)
        if not refined_data:
            raise MissingDataError

        # grab variables of interest from relevant runs
        variable_of_interest = 'time_elapsed'
        gpu_nums = [1, 4, 8]
        output_data = {}
        for img_num in self.chosen_img_nums:
            data_lists = []
            for gpu_num in gpu_nums:
                times = [entry[variable_of_interest]
                         for entry in refined_data
                         if entry['num_gpus'] == gpu_num and entry["num_images"] == img_num]
                data_lists.append(times)
            self.logger.debug(f"img_num: {img_num}, gpu_num: {gpu_num}")
            self.logger.debug(data_lists)
            output_data[str(img_num)] = self.format_data_for_error_plotting(data_lists)

        # create DataFrame from variables of interest
        # 2 is a magic number that derives from the format of
        # the data returned from format_data_for_error_plotting
        col_num = len(self.chosen_img_nums) * 2
        row_num = len(gpu_nums)
        data_array = np.zeros((row_num, col_num))
        for row in range(row_num):
            for col in range(col_num):
                var_num = col % len(self.chosen_img_nums)
                var_index = int((col - var_num) / len(self.chosen_img_nums))
                data_array[row, col] = \
                    output_data[str(self.chosen_img_nums[var_num])][var_index][row]

        self.data_df = pd.DataFrame(data_array,
                                    columns=["10000",
                                             "100000",
                                             "1000000",
                                             "10000_err",
                                             "100000_err",
                                             "1000000_err"],
                                    index=["1GPU",
                                           "4GPU",
                                           "8GPU"])

    def plot(self):
        """ Plot desired figure.

        This method first calls self.refine_data() for pare the raw data down to only the runs of
        current interest. Then, it uses matplotlib to make a scatterplot of average runtimes for a
        given number of images vs. #GPU. In the scatterplot, different numbers of images are color-
        coded.

        Outputs:
            [PDF figure.]

        """
        # pare data from all runs down to just the runs of current interest
        self.refine_data()

        # configure plot
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels
        xmin = 0.4
        xmax = 1.6
        x_ticks = [0.5, 1, 1.5]

        # plot data
        _, axis = plt.subplots(1, figsize=(10, 15))
        axis.errorbar(x_ticks,
                      self.data_df.loc[["1GPU", "4GPU", "8GPU"], "10000"],
                      yerr=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "10000_err"],
                      linestyle="None",
                      fmt='o',
                      color=self.color_maps[0](0))
        axis.errorbar(x_ticks,
                      self.data_df.loc[["1GPU", "4GPU", "8GPU"], "100000"],
                      yerr=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "100000_err"],
                      linestyle="None",
                      fmt='o',
                      color=self.color_maps[0](1))
        axis.errorbar(x_ticks,
                      self.data_df.loc[["1GPU", "4GPU", "8GPU"], "1000000"],
                      yerr=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "1000000_err"],
                      linestyle="None",
                      fmt='o',
                      color=self.color_maps[0](2))
        axis.set_title(self.plot_title)
        axis.set_ylabel(self.y_label)
        axis.set_xlabel('Number of GPUs')
        axis.set_xticks(x_ticks)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin=10, ymax=10000)
        axis.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
        axis.set_yscale("log")
        axis.legend(["10,000 Images", "100,000 Images", "1,000,000 Images"],
                    prop={'size':self.font_size})

        # save figure
        plt.savefig(path.join(self.output_folder, self.plot_pdf_name), transparent=True)
        plt.close()

        # log data
        self.logger.debug(f"Bulk runtimes for {self.chosen_delay}:")
        self.logger.debug(f'10,000 images: {self.data_df.loc[["1GPU","4GPU","8GPU"],"10000"]}')
        self.logger.debug(f'100,000 images: {self.data_df.loc[["1GPU","4GPU","8GPU"],"100000"]}')
        try:
            self.logger.debug(f"1,000,000 images: " +
                              f'{self.data_df.loc[["1GPU","4GPU","8GPU"],"1000000"]}')
        except KeyError:
            pass


class CostVsGpuFigure(BaseEmpiricalFigure):
    """ Create multiple figures of cost vs. #GPU, one for each number of images.

    This class creates multiple figures of costs for all runs with a given upload delay time,
    with each figure containing all GPU conditions for all runs with a given set of
    (delay)x(image number) conditions.

    Args:
        raw_data ([{}]): list of dictionaries, each one representing a different
            benchmarking run
        chosen_delay (float): The delay used when uploading file for the desired runs. N.B.: The
            values used indicate how many seconds our simulated uploading pipeline waited
            between simulating 100 image zip file uploads. Given that our zip files were ~150Mb
            each, the delays of 0.5s and 5.0s correspond to ~240Mbps and 2,400Mbps.
        chosen_img_num (int): the number of images processed in the desired runs
        pdf_label (str): base of title for the saved figure's PDF file
        output_folder (str): the folder into which the figures should be written
        logger_name (str, optional): the name of the class whose logger is being created

    Attributes:
        plot_title (str): title, to be used on the top of the saved matplotlib figure
        plot_pdf_name (str): output filename
        y_label (str): y-axis label for the figure; currently set to "Cost (USD)" and not
            user-configurable
        data_df (None, or pandas.DataFrame): initialized to None, but eventually used to store all
            cleaned and properly formatted data for the figures

    Outputs:
        [Sick PDF plot.]

    """
    def __init__(self, raw_data, chosen_delay, chosen_img_num, pdf_label, output_folder,
                 logger_name="CostVsGpuFigure"):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder,
                         logger_name=logger_name)
        self.plot_title = "Semantic Segmentation Costs"
        if self.chosen_delay == 5.0:
            self.plot_pdf_name = pdf_label + "_5sdelay_costs.pdf"
        elif self.chosen_delay == 0.5:
            self.plot_pdf_name = pdf_label + "_0point5sdelay_costs.pdf"
        self.y_label = "Cost (USD)"
        self.data_df = None

    def refine_data(self):
        """ Keep only data from desired runs.

        Starting with all data from all runs, pare data down to runs from the appropriate
        (delay) conditions.

        Raises:
            MissingDataError: If we don't have any data for a chosen combination of upload
                delay and image number conditions, then we raise this exception.

        """
        # only choose relevant runs
        refined_data = []
        for entry in self.raw_data:
            if entry["start_delay"] == self.chosen_delay:
                refined_data.append(entry)
        refined_data2 = []
        for entry in refined_data:
            if entry["num_images"] == self.chosen_img_num:
                refined_data2.append(entry)
        if not refined_data2:
            raise MissingDataError

        # grab variables of interest from relevant runs
        variables_of_interest = ['total_node_and_networking_costs',
                                 'cpu_node_cost',
                                 'gpu_node_cost',
                                 'extra_network_costs',
                                 'zone_egress_cost']
        gpu_nums = [1, 4, 8]
        output_data = {}
        for variable_of_interest in variables_of_interest:
            data_lists = []
            for gpu_num in gpu_nums:
                times = [entry[variable_of_interest]
                         for entry in refined_data2 if entry['num_gpus'] == gpu_num]
                data_lists.append(times)
            output_data[variable_of_interest] = self.format_data_for_error_plotting(data_lists)

        # create DataFrame from variables of interest
        # 2 is a magic number that derives from the format of
        # the data returned from format_data_for_error_plotting
        col_num = len(variables_of_interest)*2
        row_num = len(gpu_nums)
        data_array = np.zeros((row_num, col_num))
        for row in range(row_num):
            for col in range(col_num):
                var_num = col % len(variables_of_interest)
                var_index = int((col - var_num) / len(variables_of_interest))
                data_array[row, col] = output_data[variables_of_interest[var_num]][var_index][row]
        self.data_df = pd.DataFrame(data_array,
                                    columns=["total cost",
                                             "cpu node cost",
                                             "gpu node cost",
                                             "data costs",
                                             "network egress cost",
                                             "total_err",
                                             "cpu_err",
                                             "gpu_err",
                                             "data_err",
                                             "egress_err"],
                                    index=["1GPU",
                                           "4GPU",
                                           "8GPU"])

    def plot(self):
        """ Plot desired figure(s).

        This method first calls self.refine_data() for pare the raw data down to only the runs
        of current interest. Then, for each chosen (delay)x(image number) condition, it plots
        the average of the Networking, GPU node, and CPU node costs across all runs, separated
        by GPU number, as a stacked bar plot.

        Outputs:
            [PDF figure.]

        """
        # pare data from all runs down to just the runs of current interest
        self.refine_data()

        # configure plot
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels
        xmin = -0.25
        xmax = 2.25
        bar_width = 0.35
        x_ticks = [0.18, 1, 1.82]

        # create plot
        _, axis = plt.subplots(1, figsize=(15, 10))
        axis.bar(
            x_ticks,
            self.data_df.loc[["1GPU", "4GPU", "8GPU"], "data costs"],
            yerr=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "data_err"],
            width=bar_width,
            color=self.color_maps[0](0),
            bottom=None)
        axis.bar(
            x_ticks,
            self.data_df.loc[["1GPU", "4GPU", "8GPU"], "gpu node cost"],
            yerr=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "gpu_err"],
            width=bar_width,
            color=self.color_maps[0](1),
            bottom=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "data costs"])
        axis.bar(
            x_ticks,
            self.data_df.loc[["1GPU", "4GPU", "8GPU"], "cpu node cost"],
            yerr=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "cpu_err"],
            width=bar_width,
            color=self.color_maps[0](2),
            bottom=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "gpu node cost"] + \
                   self.data_df.loc[["1GPU", "4GPU", "8GPU"], "data costs"])
        axis.bar(
            x_ticks,
            self.data_df.loc[["1GPU", "4GPU", "8GPU"], "network egress cost"],
            yerr=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "egress_err"],
            width=bar_width,
            color=self.color_maps[0](3),
            bottom=self.data_df.loc[["1GPU", "4GPU", "8GPU"], "gpu node cost"] + \
                   self.data_df.loc[["1GPU", "4GPU", "8GPU"], "data costs"] + \
                   self.data_df.loc[["1GPU", "4GPU", "8GPU"], "cpu node cost"])
        axis.set_title(self.plot_title)
        axis.set_ylabel(self.y_label)
        axis.set_xlabel('Number of GPUs')
        axis.set_xticks(x_ticks)
        axis.set_xlim(xmin, xmax)
        axis.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
        axis.legend(["Data Costs", "GPU Node Cost", "CPU Node Cost", "Network Egress Cost"],
                    prop={'size':self.font_size},
                    loc=[0.30, 0.75])

        # save figure
        plt.savefig(path.join(self.output_folder, self.plot_pdf_name), transparent=True)
        plt.close()

        # brag
        self.logger.info("Created Cost Vs. GPU plot.")


class AllCostsVsGpuFigure(BaseEmpiricalFigure):
    """ Create figure of all run costs vs. #GPU.

    This class creates one chart of costs for all runs with a given upload delay. It is
    structured as a series of side-by-side, stacked bar graphs. For each number of GPUs,
    there is a cluster of bars, one for each image number, with each bar having a stack of
    data costs, gpu node costs, and cpu costs.

    Args:
        raw_data ([{}]): list of dictionaries, each one representing a different
            benchmarking run
        chosen_delay (float): The delay used when uploading file for the desired runs. N.B.:
            The values used indicate how many seconds our simulated uploading pipeline waited
            between simulating 100 image zip file uploads. Given that our zip files were ~150Mb
            each, the delays of 0.5s and 5.0s correspond to ~240Mbps and 2,400Mbps.
        chosen_img_num (int): the number of images processed in the desired runs
        output_folder (str): the folder into which the figures should be written
        logger_name (str, optional): the name of the class whose logger is being created

    Attributes:
        plot_title (str): title, to be used on the top of the saved matplotlib figure
        plot_pdf_name (str): output filename
        y_label (str): y-axis label for the figure; currently set to "Cost (USD)" and not
            user-configurable
        list_of_dfs (None, or [pandas.DataFrame]): initialized to None, but eventually used to
            store all cleaned and properly formatted data for the figures

    Outputs:
        [Sick PDF plot.]

    """
    def __init__(self, raw_data, chosen_delay, chosen_img_num, output_folder,
                 logger_name="AllCostsVsGpuFigure"):
        super().__init__(raw_data, chosen_delay, chosen_img_num, output_folder,
                         logger_name=logger_name)
        self.plot_title = "Semantic Segmentation Costs"
        if self.chosen_delay == 5.0:
            self.plot_pdf_name = "all_5sdelay_costs.pdf"
        elif self.chosen_delay == 0.5:
            self.plot_pdf_name = "all_0point5sdelay_costs.pdf"
        self.y_label = "Cost (USD)"
        self.list_of_dfs = []
        # want to make sure that we have more than one image number being passed in
        # otherwise, there's not much of a point in making the plot
        assert len(self.chosen_img_nums) > 1

    def refine_data(self):
        """ Keep only data from desired runs.

        Starting with all data from all runs, pare data down to runs from the appropriate
        (delay) conditions.

        Raises:
            MissingDataError: If we don't have any data for a chosen delay, then we raise this
                exception.

        """
        # only choose relevant runs
        refined_data = []
        for entry in self.raw_data:
            if entry["start_delay"] == self.chosen_delay:
                refined_data.append(entry)

        # construct list of data arrays for variables of interest
        variables_of_interest = ['total_node_and_networking_costs',
                                 'cpu_node_cost',
                                 'gpu_node_cost',
                                 'extra_network_costs',
                                 'zone_egress_cost']
        gpu_nums = [1, 4, 8]
        col_num = len(variables_of_interest)*2
        row_num = len(gpu_nums)
        for img_size_num in self.chosen_img_nums:
            refined_data2 = []
            for entry in refined_data:
                if entry["num_images"] == img_size_num:
                    refined_data2.append(entry)
            if not refined_data2:
                raise MissingDataError

            # grab variables of interest from relevant runs
            output_data = {}
            for variable_of_interest in variables_of_interest:
                data_lists = []
                for gpu_num in gpu_nums:
                    times = [entry[variable_of_interest]
                             for entry in refined_data2 if entry['num_gpus'] == gpu_num]
                    data_lists.append(times)
                output_data[variable_of_interest] = self.format_data_for_error_plotting(data_lists)

            # create DataFrame from variables of interest
            # 2 is a magic number that derives from the format of
            # the data returned from format_data_for_error_plotting
            data_array = np.zeros((row_num, col_num))
            for row in range(row_num):
                for col in range(col_num):
                    var_num = col % len(variables_of_interest)
                    var_index = int((col - var_num) / len(variables_of_interest))
                    data_array[row, col] = \
                        output_data[variables_of_interest[var_num]][var_index][row]
            data_df = pd.DataFrame(data_array,
                                   columns=["total cost",
                                            "cpu node cost",
                                            "gpu node cost",
                                            "data costs",
                                            "network egress cost",
                                            "total_err",
                                            "cpu_err",
                                            "gpu_err",
                                            "data_err",
                                            "egress_err"],
                                   index=["1GPU", "4GPU", "8GPU"])

            # append to the list of dataframes
            self.list_of_dfs.append(data_df)

    def plot(self):
        """ Plot desired figure.

        This method first calls self.refine_data() to pare the raw data down to a list of
        DataFrames for the runs of current interest. Then, for each DataFrame, which
        corresponds to a (delay)x(image number) condition, it plots the average of the
        Networking, GPU node, and CPU node costs across all runs, separated by GPU number,
        as a stacked bar plot. All bars for a given GPU number are clustered together to
        create a single multi-series, side-by-side stacked bar graph.

        Outputs:
            [PDF figure.]

        """
        # pare data from all runs down to just the runs of current interest
        self.refine_data()

        # confiugre plot
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels

        # create plot
        _, axis = plt.subplots(1, figsize=(15, 10))
        xmin = -0.25
        xmax = 2.25
        bar_width = 0.15
        x_ticks = [0, 1, 2]
        conditions_num = len(self.chosen_img_nums)
        all_x_ticks = []
        if conditions_num == 1:
            all_x_ticks.append(x_ticks)
        else:
            midpoint = conditions_num / 2
            for condition_index in range(conditions_num):
                difference = condition_index - midpoint
                offset = difference + 0.5
                offset_x_ticks = [x_tick + bar_width*offset for x_tick in x_ticks]
                all_x_ticks.append(offset_x_ticks)
        for img_num_index, data_df in enumerate(self.list_of_dfs):
            # have to account for the incomplete 1,000,000 image dataset
            gpu_labels = ["1GPU", "4GPU", "8GPU"]
            x_ticks_begin = 0
            x_ticks_end = None
            axis.bar(
                all_x_ticks[img_num_index][x_ticks_begin:x_ticks_end],
                data_df.loc[gpu_labels, "data costs"],
                yerr=data_df.loc[gpu_labels, "data_err"],
                width=bar_width,
                color=self.color_maps[0](0),
                bottom=None)
            axis.bar(
                all_x_ticks[img_num_index][x_ticks_begin:x_ticks_end],
                data_df.loc[gpu_labels, "gpu node cost"],
                yerr=data_df.loc[gpu_labels, "gpu_err"],
                width=bar_width,
                color=self.color_maps[0](1),
                bottom=data_df.loc[gpu_labels, "data costs"])
            axis.bar(
                all_x_ticks[img_num_index][x_ticks_begin:x_ticks_end],
                data_df.loc[gpu_labels, "cpu node cost"],
                yerr=data_df.loc[gpu_labels, "cpu_err"],
                width=bar_width,
                color=self.color_maps[0](2),
                bottom=data_df.loc[gpu_labels, "gpu node cost"] + \
                       data_df.loc[gpu_labels, "data costs"])
            axis.bar(
                all_x_ticks[img_num_index][x_ticks_begin:x_ticks_end],
                data_df.loc[gpu_labels, "network egress cost"],
                yerr=data_df.loc[gpu_labels, "egress_err"],
                width=bar_width,
                color=self.color_maps[0](3),
                bottom=data_df.loc[gpu_labels, "gpu node cost"] + \
                       data_df.loc[gpu_labels, "data costs"] + \
                       data_df.loc[gpu_labels, "cpu node cost"])
        axis.set_title(self.plot_title)
        axis.set_ylabel(self.y_label)
        axis.set_xlabel('Number of GPUs')
        axis.set_xticks(x_ticks)
        axis.set_xlim(xmin, xmax)
        axis.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
        if self.chosen_delay == 0.5:
            axis.legend(["Data Costs", "GPU Node Cost", "CPU Node Cost", "Network Egress Cost"],
                        prop={'size':self.font_size},
                        loc=[0.02, 0.80])
        elif self.chosen_delay == 5.0:
            axis.legend(["Data Costs", "GPU Node Cost", "CPU Node Cost", "Network Egress Cost"],
                        prop={'size':self.font_size},
                        loc=[0.6, 0.80])

        #save figure
        plt.savefig(path.join(self.output_folder, self.plot_pdf_name), transparent=True)
        plt.close()

        # brag
        self.logger.info("Created All Costs Vs GPU plot.")


class OptimalGpuNumberFigure(BaseFigure):
    """ Create plot of optimal GPU numbers for given netowkr and software conditions.

    This class creates a single figure that illustrates the optimal number of GPUs to choose
    for your cluster (assumed to be NVIDIA Tesla V100s), given an image upload speed and a
    model processing time. It is assumed that the user needs to process a large number of
    images, in this use case, and wants to utilize GPU hardware as efficiently as possible,
    since, in practice, that is often the single most expensive component of the cloud hardware
    used by the Deepcell Kiosk. It is further assumed that images are ~1.5Mb each. The
    resulting plot has model prediction speed (images/s) on the x-axis and ideal number of GPUs
    on the y-axis. The plotted data are lines representing different upload speeds (Mbps).

    Arguments:
        output_folder (str): The folder into which the figures should be written.
        logger_name (str, optional): the name of the class whose logger is being created

    Attributes:
        plot_pdf_name (str): output filename

    Outputs:
        [Sick PDF plot.]

    """
    def __init__(self, output_folder, logger_name="OptimalGpuNumberFigure"):
        super().__init__(output_folder, logger_name=logger_name)
        self.plot_pdf_name = "optimal_gpu_nums.pdf"

    def plot(self):
        """ Plot desired figure.

        This function plots optimal contours (for a given data transfer rate)[Mbps] on a
        (model processing speed)[images/s] vs. (ideal # GPUs)[images/s] plot. Note that, unlike
        the plot() methods of the other classes in this file, this method does not call a
        refine_data() method, because it does not utilize empirical data. Instead, we briefly
        calculate the required theoretical data below.

        """
        # declare data for plotting
        image_size_bits = 8000000 # (bits/image), 1000 pixels x 1000 pixels x 8 bits
        bits_to_megabits_conversion_factor = 1000*1000 # bits/megabit
        megabits_per_image = image_size_bits/bits_to_megabits_conversion_factor # megabits/image
        prediction_speed = np.linspace(0.1, 100, num=1000) # images/sec
        upload_speed = [10, 50, 250, 700, 1200, 1800, 2400, 3000] # Mb/s
        all_gpu_nums = np.zeros((len(prediction_speed), len(upload_speed)))
        for i, upload in enumerate(upload_speed):
            gpus = [round((upload/ (prediction*megabits_per_image)) - 0.5)
                    for prediction in prediction_speed]
            gpus = [gpu if gpu >= 1.0 else 1.0 for gpu in gpus]
            all_gpu_nums[:, i] = gpus

        # configure plot
        matplotlib.rcParams['pdf.fonttype'] = 42
        plt.rc('axes', titlesize=self.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.font_size)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.font_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.font_size)    # fontsize of the tick labels
        title = "Optimal GPU Numbers for Given Network Speeds and Tensorflow Processing Speeds"
        xlabel = "Prediction Speed (images/s)"
        ylabel = "Number of GPUs"
        xmin = 0
        xmax = max(prediction_speed)+1
        ymin = 0.98
        ymax = 8.02

        # create plot
        _, axis = plt.subplots(1, figsize=(20, 10))
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        for i, rate in enumerate(upload_speed):
            plot_label = str(rate) + " Mbps"
            color_index = i % self.color_maps[0].N
            axis.plot(
                prediction_speed,
                all_gpu_nums[:, i],
                color=self.color_maps[0](color_index),
                label=plot_label)
        legend_labels = []
        for rate in upload_speed:
            label = str(rate) + " Mbps"
            legend_labels.append(label)
        axis.legend(legend_labels, prop={'size':self.font_size})

        # save plot figure
        plt.savefig(path.join(self.output_folder, self.plot_pdf_name), transparent=True)
        plt.close()

        # brag
        self.logger.info("Created optimal GPU plot.")
