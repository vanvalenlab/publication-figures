# graph_creation.py
"""
The purpose of this script is to coordinate the production of the publication figures.
Preprint available here: https://www.biorxiv.org/content/10.1101/505032v3

For the time being, the user should customize the values in the if __name__=='__main__' block
at the bottom of the file before execution. Soon, hopefully, someone'll add argparse functionality,
so that the parameters can be passed in via the command line.

Until argparse functionality is implemented, call this file at the command line with, simply:
python3 graph_creation.py

Attributes:
    INPUT_FOLDER (str, "new_data"): folder containing raw benchmarking JSON files
    OUTPUT_FOLDER (str, "outputs"): path, absolute or relative, of folder to place output figures
        in; does not necessarily need to exist yet
    DELAYS ([float], [0.5, 5.0]): examine "start_delay" field in raw benchmarking JSON files and
        only keep those with these delays; only runs with these delays will be included in plots
    IMG_NUMS ([int], [10000, 100000, 1000000]): only runs with these numbers of images will be
        included in plots
    CREATE_IMAGE_TIME_VS_GPU_FIGURE (bool, True): create ImageTimeVsGpuFigure(s); see figures.py for
        details on this figure
    CREATE_BULK_TIME_VS_GPU_FIGURE (bool, True): create BulkTimeVsGpu(s); see figures.py for details
        on this figure
    CREATE_COST_VS_GPU_FIGURE (bool, True): create CostVsGpuFigure(s); see figures.py for details on
        this figure
    CREATE_ALL_COSTS_VS_GPU_FIGURE (bool, True): create AllCostsVsGpuFigure(s); see figures.py for
        details on this figure
    CREATE_OPTIMAL_GPU_NUMBER_FIGURE (bool, True): create OptimalGpuNumberFigure; see figures.py for
        details on this figure
    LOGGER (logging.getLogger(), logging.getLogger("GraphCreationEntrypoint")): the default logger
        for the entire module
    BASE_PATH (str, getcwd()): base path for converting relative INPUT_FOLDER and OUTPUT_FOLDER
        paths into absolute paths

Todo:
    * add argparse functionality, so users can pass in values via command line

"""


import logging
from os import path, getcwd, mkdir

from figure_generation.utils import MissingDataError
from figure_generation.figures import ImageTimeVsGpuFigure, BulkTimeVsGpuFigure, CostVsGpuFigure, \
                                  OptimalGpuNumberFigure, AllCostsVsGpuFigure
from figure_generation.data_extractor import DataExtractor


def create_figures():
    """ Create figures according to global flags.

    This function takes in the user-specified parameters and coordinates
    the creation of publication figures. Please view the `if __name__ == __main__:`
    block in the source code to view or set parameters.

    """
    create_theoretical_figures()

    if CREATE_IMAGE_TIME_VS_GPU_FIGURE or CREATE_BULK_TIME_VS_GPU_FIGURE or \
       CREATE_COST_VS_GPU_FIGURE or CREATE_ALL_COSTS_VS_GPU_FIGURE:
        raw_data = read_in_data()
        create_empirical_figures(raw_data)

def create_empirical_figures(raw_data):
    """ Create empirical (data-based) figures according to global flags.

    This method takes in the raw_data extracted from JSON files and creates all data-based
    figures.

    Args:
        raw_data ([{}]): data extracted from JSON benchmarking files

    Raises:
        TypeError: If contents of [IMG_NUMS] are not of the expected lengths. The logic here is
            that we're only expecting values of 10,000; 100,000; or 1,000,000 right now.

    """
    for chosen_delay in DELAYS:
        # create multiple-image-number graph
        if CREATE_BULK_TIME_VS_GPU_FIGURE:
            # num_gpu vs. time plot
            time_vs_gpu_figure = BulkTimeVsGpuFigure(
                raw_data, chosen_delay, IMG_NUMS, OUTPUT_FOLDER)
            try:
                time_vs_gpu_figure.plot()
                LOGGER.info(f"Saved {time_vs_gpu_figure.plot_pdf_name}")
            except MissingDataError:
                LOGGER.error(
                    f"No data for gpu vs time for "
                    f"delay {chosen_delay} and multiple image conditions.")
        if CREATE_ALL_COSTS_VS_GPU_FIGURE:
            # num_gpu vs. cost plot
            all_costs_vs_gpu_figure = AllCostsVsGpuFigure(
                raw_data, chosen_delay, IMG_NUMS, OUTPUT_FOLDER)
            try:
                all_costs_vs_gpu_figure.plot()
                LOGGER.info(f"Saved {all_costs_vs_gpu_figure.plot_pdf_name}")
            except MissingDataError:
                LOGGER.error(
                    f"No data for gpu vs cost for "
                    f"delay {chosen_delay} and img_num {chosen_img_num}.")

        # create single-image-number graphs
        for chosen_img_num in IMG_NUMS:
            logging.info(f"Delay: {chosen_delay}, number of images: {chosen_img_num}")
            chosen_img_str = str(chosen_img_num)
            if len(chosen_img_str) >= 7:
                pdf_label = chosen_img_str[:-6] + "m"
            elif len(chosen_img_str) >= 4:
                pdf_label = chosen_img_str[:-3] + "k"
            else:
                raise TypeError("Is this a number of the right length?")

            if CREATE_IMAGE_TIME_VS_GPU_FIGURE:
                # num_gpu vs. image time plot
                image_time_vs_gpu_figure = ImageTimeVsGpuFigure(
                    raw_data, chosen_delay, chosen_img_num, pdf_label, OUTPUT_FOLDER)
                try:
                    image_time_vs_gpu_figure.plot()
                    LOGGER.info(f"Saved {image_time_vs_gpu_figure.plot_pdf_name}")
                except MissingDataError:
                    LOGGER.error(
                        f"No data for gpu vs image time for "
                        f"delay {chosen_delay} and img_num {chosen_img_num}.")
            if CREATE_COST_VS_GPU_FIGURE:
                # num_gpu vs. cost plot
                cost_vs_gpu_figure = CostVsGpuFigure(
                    raw_data, chosen_delay, chosen_img_num, pdf_label, OUTPUT_FOLDER)
                try:
                    cost_vs_gpu_figure.plot()
                    LOGGER.info(f"Saved {cost_vs_gpu_figure.plot_pdf_name}")
                except MissingDataError:
                    LOGGER.error(
                        f"No data for gpu vs cost for "
                        f"delay {chosen_delay} and img_num {chosen_img_num}.")

def create_theoretical_figures():
    """ This method creates all theoretical (non-data-based) figures."""

    # trace gpu_num lines on top of a (model processing time) vs. (transfer speed) plot
    if CREATE_OPTIMAL_GPU_NUMBER_FIGURE:
        optimal_gpu_number_figure = OptimalGpuNumberFigure(OUTPUT_FOLDER)
        try:
            optimal_gpu_number_figure.plot()
            LOGGER.info(f"Saved {optimal_gpu_number_figure.plot_pdf_name}")
        except MissingDataError:
            LOGGER.error(
                f"How did we get a MissingDataError for an OptimalGpuNumberFigure plot? Lol.")

def read_in_data():
    """ This method extracts benchmarking data from all JSON files in INPUT_FOLDER.

    Args:
        INPUT_FOLDER (str; global): the folder containing the JSON benchmarking results files

    Returns:
        data_extractor.aggregated_data ([{}]): all JSON data, extracted as a list of dicts, with
            data cleaned

    """
    # extract data from json files and format it
    data_extractor = DataExtractor(INPUT_FOLDER)
    data_extractor.extract_data()
    return data_extractor.aggregated_data

def configure_logging():
    """This function configures logging for the whole module."""

    LOGGER.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    LOGGER.propagate = False
    LOGGER.addHandler(sh)
    fh = logging.FileHandler("figure_creation.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


if __name__ == '__main__':
    # configuration parameters
    INPUT_FOLDER = "new_data"
    OUTPUT_FOLDER = "outputs"
    DELAYS = [0.5, 5.0]
    IMG_NUMS = [10000, 100000, 1000000]
    CREATE_IMAGE_TIME_VS_GPU_FIGURE = True
    CREATE_BULK_TIME_VS_GPU_FIGURE = True
    CREATE_COST_VS_GPU_FIGURE = True
    CREATE_ALL_COSTS_VS_GPU_FIGURE = True
    CREATE_OPTIMAL_GPU_NUMBER_FIGURE = True

    # configure logging
    LOGGER = logging.getLogger("GraphCreationEntrypoint")
    configure_logging()

    # validate configuration parameters
    # If a folder starts with a slash, see if it exsits, and make it, if not.
    # If a folder doesn't start with a slash, see if it exists in the current directory.
    # If not, see if it exists in the parent directory. If not, create it in the parent directory.
    BASE_PATH = getcwd()
    if INPUT_FOLDER[0] != "/":
        INPUT_FOLDER = path.join(BASE_PATH, INPUT_FOLDER)
        try:
            assert path.isdir(INPUT_FOLDER)
        except AssertionError:
            BASE_PATH = '/'.join(BASE_PATH.split('/')[:-1])
            INPUT_FOLDER = path.join(BASE_PATH, INPUT_FOLDER.split('/')[-1])
            try:
                assert path.isdir(INPUT_FOLDER)
            except AssertionError:
                mkdir(INPUT_FOLDER)
    else:
        try:
            assert path.isdir(INPUT_FOLDER)
        except AssertionError:
            mkdir(INPUT_FOLDER)
    BASE_PATH = getcwd()
    if OUTPUT_FOLDER[0] != "/":
        OUTPUT_FOLDER = path.join(BASE_PATH, OUTPUT_FOLDER)
        try:
            assert path.isdir(OUTPUT_FOLDER)
        except AssertionError:
            BASE_PATH = '/'.join(BASE_PATH.split('/')[:-1])
            OUTPUT_FOLDER = path.join(BASE_PATH, OUTPUT_FOLDER.split('/')[-1])
            try:
                assert path.isdir(OUTPUT_FOLDER)
            except AssertionError:
                mkdir(OUTPUT_FOLDER)
    else:
        try:
            assert path.isdir(OUTPUT_FOLDER)
        except AssertionError:
            mkdir(OUTPUT_FOLDER)
    LOGGER.debug(f"Input directory: {INPUT_FOLDER}, Output directory: {OUTPUT_FOLDER}")

    # create figures
    create_figures()
