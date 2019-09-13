# graph_creation.py
"""
graph_creation.py

The purpose of this script is to coordinate the production of the publication figures.
"""


import logging
from os import path, getcwd

from graph_scripts import utils
from graph_scripts.utils import MissingDataError
from graph_scripts.figures import ImageTimeVsGpuFigure, BulkTimeVsGpuFigure, CostVsGpuFigure, OptimalGpuNumberFigure
from graph_scripts.data_extractor import DataExtractor

def create_figures():
    """
    This function takes in the user-specified parameters and coordinates
    the creation of publicatoin figures. Please view the `if __name__ == __main__:`
    block in the source code to view or set parameters.
    """

    create_theoretical_figures()

    if CREATE_IMAGE_TIME_VS_GPU_FIGURE or CREATE_BULK_TIME_VS_GPU_FIGURE or CREATE_COST_VS_GPU_FIGURE:
        raw_data = read_in_data()
        create_empirical_figures(raw_data)

def create_empirical_figures(raw_data):
    for chosen_delay in DELAYS:
        # create multiple-image-number graph
        if CREATE_BULK_TIME_VS_GPU_FIGURE:
            # num_gpu vs. time plot
            time_vs_gpu_figure = BulkTimeVsGpuFigure(
                raw_data, chosen_delay, IMG_NUMS, OUTPUT_FOLDER)
            try:
                time_vs_gpu_figure.plot()
                logging.debug(f"Saved {time_vs_gpu_figure.plot_pdf_name}")
            except MissingDataError:
                logging.error(
                    f"No data for gpu vs time for "
                    f"delay {chosen_delay} and multiple image conditions.")

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
                    logging.debug(f"Saved {image_time_vs_gpu_figure.plot_pdf_name}")
                except MissingDataError:
                    logging.error(
                        f"No data for gpu vs image time for "
                        f"delay {chosen_delay} and img_num {chosen_img_num}.")
            if CREATE_COST_VS_GPU_FIGURE:
                # num_gpu vs. cost plot
                cost_vs_gpu_figure = CostVsGpuFigure(
                    raw_data, chosen_delay, chosen_img_num, pdf_label, OUTPUT_FOLDER)
                try:
                    cost_vs_gpu_figure.plot()
                    logging.debug(f"Saved {cost_vs_gpu_figure.plot_pdf_name}")
                except MissingDataError:
                    logging.error(
                        f"No data for gpu vs cost for "
                        f"delay {chosen_delay} and img_num {chosen_img_num}.")

def create_theoretical_figures():
    # trace gpu_num lines on top of a (model processing time) vs. (transfer speed) plot
    if CREATE_OPTIMAL_GPU_NUMBER_FIGURE:
        optimal_gpu_number_figure = OptimalGpuNumberFigure(OUTPUT_FOLDER)
        try:
            optimal_gpu_number_figure.plot()
            logging.debug(f"Saved {optimal_gpu_number_figure.plot_pdf_name}")
        except MissingDataError:
            logging.error(
                f"How did we get a MissingDataError for an OptimalGpuNumberFigure plot?")

def read_in_data():
    # extract data from json files and format it
    data_extractor = DataExtractor(INPUT_FOLDER)
    data_extractor.extract_data()
    return data_extractor.aggregated_data

def configure_logging():
    """This function configures logging for the whole module."""

    LOGGER.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)

if __name__ == '__main__':

    # configuration parameters
    INPUT_FOLDER = "new_data"
    OUTPUT_FOLDER = "outputs"
    #DELAYS = [5.0]
    DELAYS = [0.5, 5.0]
    IMG_NUMS = [10000, 100000, 1000000]
    CREATE_IMAGE_TIME_VS_GPU_FIGURE = False
    CREATE_BULK_TIME_VS_GPU_FIGURE = False
    CREATE_COST_VS_GPU_FIGURE = False
    CREATE_OPTIMAL_GPU_NUMBER_FIGURE = True

    # configure logging
    LOGGER = logging.getLogger(__name__)

    # validate configuration parameters
    if INPUT_FOLDER[0] != "/":
        INPUT_FOLDER = path.join(getcwd(), INPUT_FOLDER)
    if OUTPUT_FOLDER[0] != "/":
        OUTPUT_FOLDER = path.join(getcwd(), OUTPUT_FOLDER)
    logging.debug(f"Input directory: {INPUT_FOLDER}, Output directory: {OUTPUT_FOLDER}")
    assert path.isdir(INPUT_FOLDER)
    assert path.isdir(OUTPUT_FOLDER)

    # create figures
    create_figures()
