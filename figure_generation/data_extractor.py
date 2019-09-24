# data_extractor.py
""" This module extracts all the Deepcell Kiosk benchmarking run data from the JSON files in a given
    folder.
"""


import re
import json
import logging
from os.path import isfile
from os import path, listdir

import numpy as np
from numpy.core._exceptions import UFuncTypeError

from .utils import NoGpuError


class DataExtractor():
    """ This class extracts all the Deepcell Kiosk benchmarking run data from the json files in a
        given folder.

    Args:
        data_folder (str): folder containing benchmarking run reuslts in JSON format
        logger_name (str, optional): the name of the class whose logger is being created

    Attributes:
        data_folder (str): absolute path to folder containing raw benchmarking data
        aggregated_data ([]): collected data from all benchmarking runs in data_folder
        total_successes (int): count of benchmarking files processed without error; intiialized to
            zero
        total_failures (int): count of benchmarking files processed with errors; intiialized to
            zero
        data_keys ([str]): partial list of dictionary keys needed from benchmarking files to compute
            quantities of interest for benchmarking runs; currently set to
            ['cpu_node_cost', 'gpu_node_cost', 'total_node_and_networking_costs', 'start_delay',
            'num_jobs', 'time_elapsed'] and not user-configurable.
        logger (logging.getLogger()): logger retrieved using logger_name

    Todo:
        * clean up the usage of data_keys: either make it more comprehensive, or do away with it

    """
    def __init__(self, data_folder, logger_name="DataExtractor"):
        self.data_folder = data_folder
        self.aggregated_data = []
        # zip files where all images succeeded
        self.total_successes = 0
        # zip files which had at least one failing image and, thus, did not complete
        self.total_failures = 0
        self.data_keys = [
            'cpu_node_cost',
            'gpu_node_cost',
            'total_node_and_networking_costs',
            'start_delay',
            'num_jobs',
            'time_elapsed']
        self.logger = self.configure_logging(logger_name)

    @staticmethod
    def configure_logging(logger_name):
        """This function configures logging for the whole instance.

        Args:
            logger_name (str): the name of the class whose logger is being created

        Returns:
            Logger corresping to given logger_name

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

    def extract_data(self):
        """ This method gets a list of JSON files from self.data_folder and passes them one-by-one
            to self.handle_individual_files and appends the returned data to a list,
            self.aggregated_data.

        """
        # extract data from json files
        data_files = [
            f for f
            in listdir(self.data_folder)
            if isfile(path.join(self.data_folder, f))
            and f.endswith(".json")]
        for data_file in data_files:
            file_path = path.join(self.data_folder, data_file)
            data_to_keep = self.handle_individual_files(data_file, file_path)
            self.aggregated_data.append(data_to_keep)
        self.logger.info(f'Total successes: {self.total_successes}')
        self.logger.info(f'Total failures: {self.total_failures}')

    def handle_individual_files(self, data_file, file_path):
        """ This method takes in a JSON filepath and extracts a plethora of intersting data fields
            from the file.

        Args:
            data_file (str): simple filename of file
            file_path (str): absolute path of file on disk

        Returns:
            data_to_keep (dict): various cost and runtime data extracted from file

        Raises:
            NoGpuError: The string "GPU" wasn't found in the passed filename (data_file). This
                either means this isn't a benchmarking run, or it is a benchmarking run done on
                a CPU. Either way, we can't plot this data with the code currently in
                figures.py.

        Todo:
            * preemptively pad all data in `for i in range(json_data['num_jobs'])` block?

        """
        with open(file_path, "r") as open_file:
            json_data = json.load(open_file)
            data_to_keep = {}
            for data_key in self.data_keys:
                if "cost" in data_key:
                    # removing string datatypes
                    data_to_keep[data_key] = float(json_data[data_key])
                elif "time_elapsed" in data_key:
                    # converting from seconds to minutes
                    data_to_keep[data_key] = json_data[data_key]/60
                else:
                    data_to_keep[data_key] = json_data[data_key]
            # adding in extra data
            num_images = len(json_data['job_data'])*100
            data_to_keep['num_images'] = num_images
            try:
                num_gpus = re.search('^([0-9]+)gpu', data_file).group(1)
            except AttributeError:
                raise NoGpuError(f"Whoops, no gpus in filename {data_file}.")
            data_to_keep['num_gpus'] = int(num_gpus)
            # replacing total cost data
            network_costs = self.extra_network_costs(
                data_to_keep['num_images'], data_to_keep['time_elapsed'])
            data_to_keep['extra_network_costs'] = network_costs
            data_to_keep['total_node_and_networking_costs'] = \
                    data_to_keep['cpu_node_cost'] + \
                    data_to_keep['gpu_node_cost'] + \
                    data_to_keep['extra_network_costs']
            # compile all file-level time components and compute some aggregate properties
            data_to_keep['total_image_time'] = 0
            data_to_keep['total_image_upload_time'] = 0
            data_to_keep['total_image_prediction_time'] = 0
            data_to_keep['total_image_postprocess_time'] = 0
            data_to_keep['total_image_download_time'] = 0
            all_total_times = []
            all_upload_times = []
            all_prediction_times = []
            all_postprocess_times = []
            all_download_times = []
            seconds_in_a_minute = 60
            for i in range(json_data['num_jobs']):
                try:
                    # gather raw times
                    total_times = json_data['job_data'][i]['total_time']
                    upload_times = json_data['job_data'][i]['upload_time']
                    prediction_times = json_data['job_data'][i]['prediction_time']
                    postprocess_times = json_data['job_data'][i]['postprocess_time']
                    download_times = json_data['job_data'][i]['download_time']
                    # append to compilation lists
                    time_lists = [
                        total_times,
                        upload_times,
                        prediction_times,
                        postprocess_times,
                        download_times]
                    all_time_lists = [
                        all_total_times,
                        all_upload_times,
                        all_prediction_times,
                        all_postprocess_times,
                        all_download_times]
                    for time_list, all_time_list in zip(time_lists, all_time_lists):
                        for entry in time_list:
                            all_time_list.append(entry)
                    # increment sums
                    data_to_keep['total_image_time'] += \
                            sum(total_times)/seconds_in_a_minute
                    data_to_keep['total_image_upload_time'] += \
                            sum(upload_times)/seconds_in_a_minute
                    data_to_keep['total_image_prediction_time'] += \
                            sum(prediction_times)/seconds_in_a_minute
                    data_to_keep['total_image_postprocess_time'] += \
                            sum(postprocess_times)/seconds_in_a_minute
                    data_to_keep['total_image_download_time'] += \
                            sum(download_times)/seconds_in_a_minute
                    # register a success
                    self.total_successes += 1
                except TypeError:
                    self.logger.error(f"There was some sort of error with job number {i}" +
                                      f" in data file {data_file}.")
                    self.total_failures += 1
            data_to_keep['average_image_time'] = \
                    data_to_keep['total_image_time']/self.total_successes
            data_to_keep['average_image_upload_time'] = \
                data_to_keep['total_image_upload_time']/self.total_successes
            data_to_keep['average_image_prediction_time'] = \
                data_to_keep['total_image_prediction_time']/self.total_successes
            data_to_keep['average_image_postprocess_time'] = \
                data_to_keep['total_image_postprocess_time']/self.total_successes
            data_to_keep['average_image_download_time'] = \
                data_to_keep['total_image_download_time']/self.total_successes
            data_to_keep['all_total_times'] = all_total_times
            # this next computation is so difficult I gave it its own function
            data_to_keep['all_network_times'] = \
                self.handle_network_time_computation(all_upload_times, all_download_times)
            data_to_keep['all_upload_times'] = all_upload_times
            data_to_keep['all_prediction_times'] = all_prediction_times
            data_to_keep['all_postprocess_times'] = all_postprocess_times
            data_to_keep['all_download_times'] = all_download_times
            return data_to_keep

    def handle_network_time_computation(self, all_upload_times, all_download_times):
        """ This method handles the details of computing the 'network time', which is determined
            by a formula involving both the upload and download times during benchmarking. For now,
            we're just using 2*upload + 2*download, but we could make this a little more precise.
            I think this method gets called with image-level data for all images in all jobs in a
            run all at once, but I'm not certain.

        Args:
            all_upload_times (ndarray): the upload times observed for each image in the run
            all_download_times (ndarray): the download times observed for each image in the run

        Returns:
            all_network_times (ndarray): the 'network times' for each image in the run

        Raises:
            ValueError: If the np.add or np.multiply commands in the computation of
                all_network_times raise a ValueError, and it isn't due to length mismatches
                between all_upload_times and all_download_times, then we reraise the
                ValueError.

        Todo:
            * In the middle of a data series, we'll sometimes get four consecutive entries that
                read as ["N", "o", "n", "e"]. All four of these likely represent one missing
                numeric data point. Right now, all four are being replaced with average value
                data, but ideally they would all be consolidated into one numeric data point.
                Further, this irregularity may be the only common cause of the UFuncTypeError.
            *  Pass in a file name and a job number to this function, to improve error text in
                the `self.logger.error("We hit a ufunktypeerror!!!")` line.

        """
        upload_download_multiplier = 2
        # We need to wrap all this in a while loop because we might
        # hit multiple exceptions in series.
        while True:
            try:
                all_network_times = \
                    np.add(
                        np.dot(upload_download_multiplier, all_upload_times),
                        np.dot(upload_download_multiplier, all_download_times)
                    )
                break
            # Sometimes the upload and download series would have
            # slightly different numbers of entries. This is likely due to
            # Redis not yet having the data on rare occasions when we ask for it.
            # We don't consider this to be a huge concern and note
            # that it only marginally affects the completeness of our data.
            # Further, it doesn't bias our results.
            except ValueError:
                upload_length = len(all_upload_times)
                download_length = len(all_download_times)
                if upload_length < download_length:
                    upload_average = np.average(all_upload_times)
                    length_diff = download_length - upload_length
                    for i in range(length_diff):
                        all_upload_times = np.append(all_upload_times, upload_average)
                elif download_length < upload_length:
                    download_average = np.average(all_download_times)
                    length_diff = upload_length - download_length
                    for i in range(length_diff):
                        all_download_times = np.append(all_download_times, download_average)
                else:
                    raise ValueError
            except UFuncTypeError:
                for i, upload in enumerate(all_upload_times):
                    if isinstance(upload, str):
                        all_upload_times[i] = np.nan
                upload_mean = np.nanmean(all_upload_times)
                for i, upload in enumerate(all_upload_times):
                    if np.isnan(upload):
                        all_upload_times[i] = upload_mean
                for i, download in enumerate(all_download_times):
                    if isinstance(download, str):
                        all_download_times[i] = np.nan
                download_mean = np.nanmean(all_download_times)
                for i, download in enumerate(all_download_times):
                    if np.isnan(download):
                        all_download_times[i] = download_mean
                self.logger.error("We hit a ufunktypeerror!!!")
        return all_network_times

    @staticmethod
    def extra_network_costs(img_num, run_duration_minutes):
        """ This method computes network and storage costs during the run due to Google Cloud
            storage charges.

        Args:
            img_num (int): number of images uploaded in the current run
            run_duration_minutes (float): time from beginning to end of run

        Returns:
            total_fees (float): total fees imposed by Google Cloud over the life of the run

        """
        total_storage_gb = 1.5*img_num/1000
        run_duration_months = run_duration_minutes/24/30

        total_storage_cost = 0.026*total_storage_gb*run_duration_months
        download_fees = 0.004*img_num/10000
        publication_fees = 0.05*img_num/10000

        total_fees = total_storage_cost + download_fees + publication_fees
        return total_fees
