# graph_creation.py

# identify list of files
# define data structures
# loop:
# 1) read in json file (pyjson or something?)
# 2) extract relevant fields
# loop:
# 1) pass data structures off to each image creation function, as appropriate


import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pdb

from graph_scripts.utils import NoGpuError, MissingDataError, define_colorblind_color_map, format_data_for_error_plotting, extract_data
from graph_scripts.figures import ImageTimeVsGpu

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

if __name__=='__main__':
    cm = define_colorblind_color_map()

    extracted_data = extract_data()
    raw_data = extracted_data

    delays = [5.0]
    #delays = [0.5, 5.0]
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

    if False:
        pdb.set_trace()
