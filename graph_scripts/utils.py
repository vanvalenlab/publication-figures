# utils.py
import re
import json
import numpy as np
from os import listdir, path
from os.path import isfile
from numpy.core._exceptions import UFuncTypeError

# custom execption for a failing regex
class NoGpuError(Exception):
    pass

# custom execption for missing data
class MissingDataError(Exception):
    pass

# https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def extract_data(data_folder):
    data_files = [f for f in listdir(data_folder) if isfile(path.join(data_folder, f)) and f.endswith(".json")]

    # extract data from json files
    data_keys = ['cpu_node_cost', 'gpu_node_cost', 'total_node_and_networking_costs', 'start_delay', 'num_jobs', 'time_elapsed']
    aggregated_data = []
    total_successes = 0 # these are zip files where all images succeeded
    total_failures = 0 # these are zip files which had at least one failing image and, thus, did not complete
    for data_file in data_files:
        file_path = path.join(data_folder, data_file)
        with open(file_path, "r") as open_file:
            json_data = json.load(open_file)
            data_to_keep = {}
            for data_key in data_keys:
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
            network_costs = extra_network_costs(data_to_keep['num_images'], data_to_keep['time_elapsed'])
            data_to_keep['extra_network_costs'] = network_costs
            data_to_keep['total_node_and_networking_costs'] = \
                    data_to_keep['cpu_node_cost'] + \
                    data_to_keep['gpu_node_cost'] + \
                    data_to_keep['extra_network_costs']
            # compile all file-level time components and compute some aggregate properties
            total_time = 0
            total_upload_time = 0
            total_prediction_time = 0
            total_postprocess_time = 0
            total_download_time = 0
            all_total_times = []
            all_upload_times = []
            all_prediction_times = []
            all_postprocess_times = []
            all_download_times = []
            for i in range(json_data['num_jobs']):
                try:
                    # TODO: potentially pad all data?
                    # gather raw times
                    total_times = json_data['job_data'][i]['total_time']
                    upload_times = json_data['job_data'][i]['upload_time']
                    prediction_times = json_data['job_data'][i]['prediction_time']
                    postprocess_times = json_data['job_data'][i]['postprocess_time']
                    download_times = json_data['job_data'][i]['download_time']
                    # append to compilation lists
                    time_lists = [total_times, upload_times, prediction_times, postprocess_times, download_times]
                    all_time_lists = [all_total_times, all_upload_times, all_prediction_times, all_postprocess_times, all_download_times]
                    for time_list, all_time_list in zip(time_lists, all_time_lists):
                        for entry in time_list:
                            all_time_list.append(entry)
                    # increment sums
                    total_time = total_time + sum(total_times)
                    total_upload_time = total_upload_time + sum(upload_times)
                    total_prediction_time = total_prediction_time + sum(prediction_times)
                    total_postprocess_time = total_postprocess_time + sum(postprocess_times)
                    total_download_time = total_download_time + sum(download_times)
                    # register a success
                    total_successes = total_successes + 1
                except TypeError:
                    print("Error.")
                    total_failures = total_failures + 1
            data_to_keep['total_image_time'] = total_time/60
            data_to_keep['total_image_upload_time'] = total_upload_time/60
            data_to_keep['total_image_prediction_time'] = total_prediction_time/60
            data_to_keep['total_image_postprocess_time'] = total_postprocess_time/60
            data_to_keep['total_image_download_time'] = total_download_time/60
            data_to_keep['average_image_time'] = data_to_keep['total_image_time']/total_successes
            data_to_keep['average_image_upload_time'] = data_to_keep['total_image_upload_time']/total_successes
            data_to_keep['average_image_prediction_time'] = data_to_keep['total_image_prediction_time']/total_successes
            data_to_keep['average_image_postprocess_time'] = data_to_keep['total_image_postprocess_time']/total_successes
            data_to_keep['average_image_download_time'] = data_to_keep['total_image_download_time']/total_successes
            data_to_keep['all_total_times'] = all_total_times
            upload_download_multiplier = 2
            # We need to wrap all this in a while loop because we might hit multiple exceptions in series.
            while True:
                try:
                    data_to_keep['all_network_times'] = np.add( np.dot(upload_download_multiplier,all_upload_times), np.dot(upload_download_multiplier,all_download_times))
                    break
                # Sometimes the upload and download series would have slightly different numbers of entries.
                # This is likely due to Redis not yet having the data on rare occasions when we ask for it.
                # We don't consider this to be a huge concern and note that it only marginally affects the
                # completeness of our data. Further, it doesn't bias our results.
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
                # We had some entries that were text??? Very weird and I want to make sure this wasn't an issue.
                except UFuncTypeError as e:
                    for i, upload in enumerate(all_upload_times):
                        if type(upload)==str:
                            all_upload_times[i] = np.nan
                    upload_mean = np.nanmean(all_upload_times)
                    for i, upload in enumerate(all_upload_times):
                        if np.isnan(upload):
                            all_upload_times[i] = upload_mean
                    for i, download in enumerate(all_download_times):
                        if type(download)==str:
                            all_download_times[i] = np.nan
                    download_mean = np.nanmean(all_download_times)
                    for i, download in enumerate(all_download_times):
                        if np.isnan(download):
                            all_download_times[i] = download_mean
                    print("ufunktypeerror!!!")
            data_to_keep['all_upload_times'] = all_upload_times
            data_to_keep['all_prediction_times'] = all_prediction_times
            data_to_keep['all_postprocess_times'] = all_postprocess_times
            data_to_keep['all_download_times'] = all_download_times
            aggregated_data.append(data_to_keep)
    print(f'Total successes: {total_successes}')
    print(f'Total failures: {total_failures}')
    return aggregated_data

def extra_network_costs(img_num, run_duration_minutes):
    total_storage_gb = 1.5*img_num/1000
    run_duration_months = run_duration_minutes/24/30

    total_storage_cost = 0.026*total_storage_gb*run_duration_months
    download_fees = 0.004*img_num/10000
    publication_fees = 0.05*img_num/10000

    total_fees = total_storage_cost + download_fees + publication_fees
    return total_fees
