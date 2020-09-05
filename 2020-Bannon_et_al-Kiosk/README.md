The DeepCell Kiosk paper [(Bannon et al., 2020)](<https://www.biorxiv.org/content/10.1101/505032v3>) presents cost and runtime benchmarks for running a generic image segmentation pipeline on datasets of given sizes inside the DeepCell Kiosk with different sets of constraints (Fig. 1b). This repository exists to aid DeepCell Kiosk users in recreating these figures.
 
There are two steps to figure recreation:
1) Generating a full battery of benchmark data for the DeepCell Kiosk under a variety of conditions
2) Creating the figures from this benchmarking data

## Generating Benchmarking Data

This repository can produce the benchmarking figures from the paper, but it expects a full battery of benchmarking output files from all run conditions. It will not produce any figures as-is without output from all runs. 
 
To generate a full battery of benchmarking run data (i.e., varying image numbers and GPU numbers), please complete the following benchmarking runs:
 
* 3 runs with 1 GPU and 10,000 images
* 3 runs with 4 GPUs and 10,000 images
* 3 runs with 8 GPUs and 10,000 images
* 3 runs with 1 GPU and 100,000 images
* 3 runs with 4 GPUs and 100,000 images
* 3 runs with 8 GPUs and 100,000 images
* 3 run with 1 GPU and 1,000,000 images
* 3 run with 4 GPUs and 1,000,000 images
* 3 run with 8 GPUs and 1,000,000 images
 
To do this, follow the instructions in [Developer docs of the kiosk-console repo](https://deepcell-kiosk.readthedocs.io/en/master/DEVELOPER.html#benchmarking-the-deepcell-kiosk), in addition to the constraints outlined below in the `Benchmarking Data Settings` subsection.
 
### Benchmarking Data Settings
 
When recreating the figures from the DeepCell Kiosk paper, please observe the configuration guidelines below. If you'd like to see the exact configuration the Van Valen Lab used for benchmarking, feel free to checkout the `benchmarks` branch of the `kiosk-console` repository.
 
* In the DeepCell Kiosk paper, benchmarking data was presented for clusters with maxima of 1, 4, and 8 GPUs. Choose the appropriate maximum during DeepCell Kiosk configuration for the benchmarking dataset you would like to recreate
* Keep the default values for `MODEL` and `FILE` in the benchmarking YAML file, `conf/helmfile.d/0410.benchmarking.yaml` in the `kiosk-console` repository.
* Copy the `models/NuclearSegmentation` folder in the Van Valen Lab's [kiosk-benchmarking bucket](https://console.cloud.google.com/storage/browser/kiosk-benchmarking) to the same location in your benchmarking bucket
* Copy the Van Valen Lab's [zip100.zip](https://console.cloud.google.com/storage/browser/_details/kiosk-benchmarking/sample-data/zip100.zip) into the `uploads` folder of your benchmarking bucket. This file consists of 100 microscopy images, which we used as the basis for all the benchmarking runs in the DeepCell Kiosk paper
* Benchmarking data was presented in the DeepCell Kiosk paper for 10,000-image, 100,000-image, and 1,000,000-image runs. Since `zip100.zip` contains 100 images, set the `COUNT` variable in the benchmarking YAML file to either 100, 1,000, or 10,000

## Figure Creation

1) To create the figures, first install the `deepcell-kiosk-figure-generation` package using the included `setup.py` file with the command `pip install -e .`
2) Place all JSON benchmarking data in a top-level folder inside the project. (An empty `new_data` folder is included for this purpose, but feel free to use any folder name you like.)
3) Edit the global attributes in `figure_generation/graph_creation.py` to suit your tastes. (The global attributes should all be documented at the top of `figure_generation/graph_creation.py`.)
4) Execute `python3 ./figure_generation/graph_creation.py`

Note that we used Illustrator to make aesthetic improvements to the figures generated from this repository.
