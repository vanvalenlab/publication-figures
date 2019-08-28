# utils.py

# custom execption for missing data
class MissingDataError(Exception):
    pass

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

def define_colorblind_color_map(self):
    # (R,G,B) from nature methods Sky blue, bluish green, reddish purple, vermillion, orange
    colors = [(86/255,180/255,233/255), (0,158/255,115/255), (204/255,121/255,167/255), (213/255,94/255,0), (230/255,159/255,0)]

    n_bins = 5
    cmap_name = 'colorblind'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cmap

# https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

