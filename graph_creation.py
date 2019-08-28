# graph_creation.py

# identify list of files
# define data structures
# loop:
# 1) read in json file (pyjson or something?)
# 2) extract relevant fields
# loop:
# 1) pass data structures off to each image creation function, as appropriate


import json
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pdb

from graph_scripts.utils import MissingDataError
from graph_scripts.figures import ImageTimeVsGpu

# custom execption for a failing regex
class NoGpuError(Exception):
    pass

# custom execption for missing data
#class MissingDataError(Exception):
#    pass

# utility function:
# https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def extract_data():
    data_dir = "/home/linus/projects/vanvalen_kiosk/figure-generation/new_data"
    data_files = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith(".json")]

    # extract data from json files
    data_keys = ['cpu_node_cost', 'gpu_node_cost', 'total_node_and_networking_costs', 'start_delay', 'num_jobs', 'time_elapsed']
    aggregated_data = []
    total_successes = 0 # these are zip files where all images succeeded
    total_failures = 0 # these are zip files which had at least one failing image and, thus, did not complete
    for data_file in data_files:
        file_path = join(data_dir, data_file)
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
            # computing aggregate file-level time components
            total_time = 0
            total_upload_time = 0
            total_prediction_time = 0
            total_postprocess_time = 0
            total_download_time = 0
            for i in range(100):
                try:
                    total_time = total_time + sum(json_data['job_data'][i]['total_time'])
                    total_upload_time = total_upload_time + sum(json_data['job_data'][i]['upload_time'])
                    total_prediction_time = total_prediction_time + sum(json_data['job_data'][i]['prediction_time'])
                    total_postprocess_time = total_postprocess_time + sum(json_data['job_data'][i]['postprocess_time'])
                    total_download_time = total_download_time + sum(json_data['job_data'][i]['download_time'])
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


def get_time_by_gpu_data(raw_data, chosen_delay, img_nums):
    # only choose relevant runs
    refined_data = []
    for entry in raw_data:
        if entry["start_delay"] == chosen_delay:
            refined_data.append(entry)
    if len(refined_data) == 0:
        raise MissingDataError
    # grab variables of interest from relevant runs
    variable_of_interest = 'time_elapsed'
    gpu_nums = [1,4,8]
    output_data = {}
    for img_num in img_nums:
        data_lists = []
        for gpu_num in gpu_nums:
            times = [entry[variable_of_interest] for entry in refined_data if entry['num_gpus']==gpu_num and entry["num_images"]==img_num]
            data_lists.append(times)
        print(f"img_num: {img_num}, gpu_num: {gpu_num}")
        print(data_lists)
        output_data[str(img_num)] = format_data_for_error_plotting(data_lists)
    # create DataFrame from variables of interest
    # 2 is a magic number that derives from the format of the data returned from format_data_for_error_plotting
    col_num = len(img_nums)*2
    row_num = len(gpu_nums)
    data_array = np.zeros( (row_num, col_num) )
    for row in range(row_num):
        for col in range(col_num):
            var_num = col % len(img_nums)
            var_index = int( (col - var_num) / len(img_nums) )
            data_array[row,col] = output_data[ str(img_nums[var_num]) ][ var_index ][ row ]
    data_df = pd.DataFrame(data_array, columns=["10000","100000","10000_err","100000_err"], index=["1GPU","4GPU","8GPU"])
    return data_df


def get_cost_by_gpu_data(raw_data, chosen_delay, chosen_img_num):
    # generate df
    refined_data = []
    for entry in raw_data:
        if entry["start_delay"] == chosen_delay:
            refined_data.append(entry)
    refined_data2 = []
    for entry in refined_data:
        if entry["num_images"] == chosen_img_num:
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
        output_data[variable_of_interest] = format_data_for_error_plotting(data_lists)
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
    data_df = pd.DataFrame(data_array, columns=["total cost", "cpu node cost","gpu node cost","network costs","total_err","cpu_err","gpu_err","network_err"], index=["1GPU","4GPU","8GPU"])
    return data_df


def format_data_for_error_plotting(data_lists):
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


def plot_times(df, labels=None, title="multiple unstacked bar plot", y_label=None, H="/", pdf_name="output-bargraph.pdf", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
       labels is a list of the names of the dataframe, used for the legend title is a string for the 
       title of the plot H is the hatch used for identification of the different dataframe"""

    n_col = len(df.columns) 
    fig, axes = plt.subplots(1, figsize=(20,10))
    axe = axes
    axe = df[["10000", "100000"]].plot(kind="line",
                    yerr=df[["10000_err", "100000_err"]].values.T,
                    linewidth=2,
                    stacked=False,
                    ax=axe,
                    legend=False,
                    grid=False,
                    **kwargs)  # make bar plots

    axe.set_title(title)
    axe.set_ylabel(y_label)
    axe.set_xlabel('Number of GPUs')
    axe.set_xticks([0,1,2])
    axe.set_xlim(-0.05,2.05)
    axe.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    l1 = axe.legend(h[:n_col], l[:n_col], loc=[.425, 0.75])
    axe.add_artist(l1)
    plt.savefig(pdf_name, transparent=True)
    return axe

def plot_costs(df, labels=None, title="multiple unstacked bar plot", y_label=None, H="/", pdf_name="output-bargraph.pdf", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
       labels is a list of the names of the dataframe, used for the legend title is a string for the 
       title of the plot H is the hatch used for identification of the different dataframe"""

    n_col = len(df.columns) 
    fig, axes = plt.subplots(1, figsize=(20,10))
    axe = axes
    axe = df[["total cost", "cpu node cost", "gpu node cost", "network costs"]].plot(kind="line",
                    yerr=df[["total_err", "cpu_err", "gpu_err", "network_err"]].values.T,
                    linewidth=2,
                    stacked=False,
                    ax=axe,
                    legend=False,
                    grid=False,
                    **kwargs)  # make bar plots

    axe.set_title(title)
    axe.set_ylabel(y_label)
    axe.set_xlabel('Number of GPUs')
    axe.set_xticks([0,1,2])
    axe.set_xlim(-0.05,2.05)
    axe.set_xticklabels(["1 GPU", "4 GPU", "8 GPU"])
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    l1 = axe.legend(h[:n_col], l[:n_col], loc=[.425, 0.75])
    axe.add_artist(l1)
    plt.savefig(pdf_name, transparent=True)
    return axe

def define_colorblind_color_map():
    colors = [(86/255,180/255,233/255), (0,158/255,115/255), (204/255,121/255,167/255), (213/255,94/255,0), (230/255,159/255,0)]  # (R,G,B) from nature methods Sky blue, bluish green, reddish purple, vermillion, orange
    n_bins = 5
    cmap_name = 'colorblind'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm

if __name__=='__main__':
    cm = define_colorblind_color_map()

    extracted_data = extract_data()
    raw_data = extracted_data


    delays = [5.0]
    img_nums = [10000, 100000, 1000000]
    create_image_time_vs_gpu = True
    create_time_vs_gpu = True
    create_cost_vs_gpu = True


    for chosen_delay in delays:
        # create multiple-image-number graph
        if create_time_vs_gpu:
            # num_gpu vs. time plot
            time_by_gpu_title = "Semantic Segmentation Runtimes"
            time_by_gpu_pdf_name = "many_image_runtimes.pdf"
            try:
                df = get_time_by_gpu_data(extracted_data, chosen_delay, [10000,100000])
                plot_times(df, title=time_by_gpu_title, y_label="Time (Minutes)", pdf_name=time_by_gpu_pdf_name, cmap=cm)
                print(f"Saved {time_by_gpu_pdf_name}")
            except MissingDataError:
                print(f"No data for gpu vs time for delay {chosen_delay} and multiple image conditions.")
                pass
        
        for chosen_img_num in img_nums:
            print(f"Delay: {chosen_delay}, number of images: {chosen_img_num}")

            chosen_img_str = str(chosen_img_num)
            if len(chosen_img_str)>=7:
                pdf_label = chosen_img_str[:-6] + "m"
                title_label = chosen_img_str[:-6] + "," + chosen_img_str[-6:-3] + "," + chosen_img_str[-3:]
            elif len(chosen_img_str)>=4:
                pdf_label = chosen_img_str[:-3] + "k"
                title_label = chosen_img_str[:-3] + "," + chosen_img_str[-3:]
            else:
                raise TypeError("Is this a number of the right length?")

            # create single-image-number graphs
            if create_image_time_vs_gpu:
                # num_gpu vs. image time plot
                image_time_vs_gpu_figure = ImageTimeVsGpu(raw_data, chosen_delay, chosen_img_num, title_label, pdf_label)
                try:
                    image_time_vs_gpu_figure.plot()
                    print(f"Saved {image_time_vs_gpu_figure.plot_pdf_name}")
                except MissingDataError:
                    print(f"No data for gpu vs image time for delay {chosen_delay} and img_num {chosen_img_num}.")
                    pass
            if create_cost_vs_gpu:
                # num_gpu vs. cost plot
                cost_by_gpu_title = title_label + " Image Semantic Segmentation Cost"
                cost_by_gpu_pdf_name = pdf_label + "_costs.pdf"
                try:
                    df = get_cost_by_gpu_data(extracted_data, chosen_delay, chosen_img_num)
                    plot_costs(df, labels=["cpu node costs", "gpu node costs", "networking costs"], title=cost_by_gpu_title, y_label="Cost (US Dollars)", pdf_name=cost_by_gpu_pdf_name, cmap=cm)
                    print(f"Saved {cost_by_gpu_pdf_name}")
                except MissingDataError:
                    print(f"No data for gpu vs cost for delay {chosen_delay} and img_num {chosen_img_num}.")
                    pass
    pdb.set_trace()






    if False:
        # create single-image-number graphs
        if create_image_time_vs_gpu:
            pass
        if create_cost_vs_gpu:
            # num_gpu vs. cost plot
            try:
                cost_vs_gpu_figure = CostVsGpu(raw_data, chosen_delay, chosen_img_num, title_label, pdf_label)
                cost_vs_gpu_figure.plot()
                print(f"Saved {cost_vs_gpu_figure.plot_pdf_name}")
            except MissingDataError:
                print(f"No data for gpu vs cost for delay {chosen_delay} and img_num {chosen_img_num}.")
                pass
    pdb.set_trace()
