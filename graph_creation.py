# graph_creation.py

import logging
from os import path, getcwd

from graph_scripts import utils
from graph_scripts.utils import MissingDataError
from graph_scripts.figures import ImageTimeVsGpuFigure, BulkTimeVsGpuFigure, CostVsGpuFigure

def main():
    raw_data = utils.extract_data(input_folder)

    for chosen_delay in delays:
        # create multiple-image-number graph
        if create_bulk_time_vs_gpu_figure:
            # num_gpu vs. time plot
            time_vs_gpu_figure = BulkTimeVsGpuFigure(raw_data, chosen_delay, img_nums, output_folder)
            try:
                time_vs_gpu_figure.plot()
                logging.debug(f"Saved {time_vs_gpu_figure.plot_pdf_name}")
            except MissingDataError:
                logging.error(f"No data for gpu vs time for delay {chosen_delay} and multiple image conditions.")
        
        # create single-image-number graphs
        for chosen_img_num in img_nums:
            logging.info(f"Delay: {chosen_delay}, number of images: {chosen_img_num}")
            chosen_img_str = str(chosen_img_num)
            if len(chosen_img_str)>=7:
                pdf_label = chosen_img_str[:-6] + "m"
            elif len(chosen_img_str)>=4:
                pdf_label = chosen_img_str[:-3] + "k"
            else:
                raise TypeError("Is this a number of the right length?")

            if create_image_time_vs_gpu_figure:
                # num_gpu vs. image time plot
                image_time_vs_gpu_figure = ImageTimeVsGpuFigure(raw_data, chosen_delay, chosen_img_num, pdf_label, output_folder)
                try:
                    image_time_vs_gpu_figure.plot()
                    logging.debug(f"Saved {image_time_vs_gpu_figure.plot_pdf_name}")
                except MissingDataError:
                    logging.error(f"No data for gpu vs image time for delay {chosen_delay} and img_num {chosen_img_num}.")
            if create_cost_vs_gpu_figure:
                # num_gpu vs. cost plot
                cost_vs_gpu_figure = CostVsGpuFigure(raw_data, chosen_delay, chosen_img_num, pdf_label, output_folder)
                try:
                    cost_vs_gpu_figure.plot()
                    logging.debug(f"Saved {cost_vs_gpu_figure.plot_pdf_name}")
                except MissingDataError:
                    logging.error(f"No data for gpu vs cost for delay {chosen_delay} and img_num {chosen_img_num}.")

if __name__=='__main__':

    # configuration parameters
    input_folder = "new_data"
    output_folder = "outputs"
    delays = [5.0]
    #delays = [0.5, 5.0]
    img_nums = [10000, 100000, 1000000]
    create_image_time_vs_gpu_figure = True
    create_bulk_time_vs_gpu_figure = True
    create_cost_vs_gpu_figure = True

    # configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # validate configuration parameters
    if input_folder[0]=="/":
        input_folder = input_folder
    else:
        input_folder = path.join(getcwd(),input_folder)
    if output_folder[0]=="/":
        output_folder = output_folder
    else:
        output_folder = path.join(getcwd(),output_folder)
    logging.debug(f"Input directory: {input_folder}, Output directory: {output_folder}")
    assert path.isdir(input_folder)
    assert path.isdir(output_folder)

    # create plots
    main()
    
    if False:
        import pdb; pdb.set_trace()
