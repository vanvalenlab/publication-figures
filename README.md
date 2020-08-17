[![Documentation Status](https://readthedocs.org/projects/deepcell-kiosk-figure-generation/badge/?version=latest)](https://deepcell-kiosk-figure-generation.readthedocs.io/en/latest/?badge=latest)

The DeepCell Kiosk paper [(Bannon et al., 2020)](<https://www.biorxiv.org/content/10.1101/505032v3>) presents cost and runtime benchmarks for running a generic image segmentation pipeline on datasets of given sizes inside the DeepCell Kiosk with different sets of constraints (Fig. 1b).
 
N.B.: Before following any of these steps, I'd recommend activating a Python `virtualenv`. It won't affect how easy it is to run this project, probably, but it will make your life easier after you've forgotten about this project and moved on to other Python packages.

(To view API documentation for this project online, please go [here](https://deepcell-kiosk-figure-generation.readthedocs.io).)

## Figure Creation
1) To create the figures, first install the `deepcell-kiosk-figure-generation` package using the included `setup.py` file with the command `pip install -e .`.
2) Place all JSON benchmarking data in a top-level folder inside the project. (The project comes with a `new_data` folder containing default data for this purpose, but feel to throw this data away and replace it with your own, or put yours in an entirely new folder.) (Also, see the section `Regenerating DeepCell Kiosk Paper Benchmarking Data` for help on recreating the figures from the DeepCell Kiosk paper.)
3) Edit the global attributes in `figure_generation/graph_creation.py` to suit your tastes. (The global attributes should all be documented at the top of `figure_generation/graph_creation.py`.)
4) Execute `python3 ./figure_generation/graph_creation.py`.

Note that we did a significant amount of aesthetic primping in Illustrator to the figures generated from the included default data before we put them in the paper, but the data is all the same.

## Regenerating DeepCell Kiosk Paper Benchmarking Data

This repository can produce the benchmarking figures from the paper, but it expects a full battery of benchmarking output files from all run conditions. It will not produce any figures as-is without output from all runs. 
 
To generate a full battery of benchmarking run data (i.e., varying image numbers and GPU numbers), please complete the following benchmarking runs:
 
     - 3 runs with 1 GPU and 10,000 images
     - 3 runs with 4 GPUs and 10,000 images
     - 3 runs with 8 GPUs and 10,000 images
     - 3 runs with 1 GPU and 100,000 images
     - 3 runs with 4 GPUs and 100,000 images
     - 3 runs with 8 GPUs and 100,000 images
     - 3 run with 1 GPU and 1,000,000 images
     - 3 run with 4 GPUs and 1,000,000 images
     - 3 run with 8 GPUs and 1,000,000 images
 
Then, follow the instructions in [Developer docs of the kiosk-console repo](https://deepcell-kiosk.readthedocs.io/en/master/DEVELOPER.html), in addition to the constraints outlined below in the `Benchmarking Figure Recreation Settings` section, to recreate the DeepCell Kiosk paper's benchmarking figures.
 
### Benchmarking Figure Recreation Settings
 
When recreating the figures from the DeepCell Kiosk paper, please observe the following configuration guidelines:
 
    - In the DeepCell Kiosk paper, benchmarking data was presented for clusters with maxima of 1, 4, and 8 GPUs. Choose the appropriate maximum during DeepCell Kiosk configuration for the benchmarking dataset you would like to recreate.
    - Keep the default values for `MODEL` and `FILE` in the benchmarking YAML file.
    - Copy the `models/NuclearSegmentation` folder in the Van Valen Lab's [kiosk-benchmarking bucket](https://console.cloud.google.com/storage/browser/kiosk-benchmarking) to the same location in your benchmarking bucket.
    - Copy the Van Valen Lab's [zip100.zip](https://console.cloud.google.com/storage/browser/_details/kiosk-benchmarking/sample-data/zip100.zip) into the `uploads` folder of your benchmarking bucket. This file consists of 100 microscopy images, which we used as the basis for all the benchmarking runs in the DeepCell Kiosk paper.
    - Benchmarking data was presented in the DeepCell Kiosk paper for 10,000-image, 100,000-image, and 1,000,000-image runs. Since `zip100.zip` contains 100 images, set the `COUNT` variable in the benchmarking YAML file to either 100, 1,000, or 10,000.

## Miscellanea

### Documentation
If you want to create a local copy of the documentation for this project (instead of viewing the online version, linked above), follow these steps:
1) First, install the package with the `docs` dependencies included: `pip install -e .[docs]`. (It's ok if you've already installed without the docs dependencies; there won't be any conflict if you use this command after that.)
2) Move into the `docs` directory.
3) Execute `sphinx-build -b html . ./_build/`. Assuming no errors were encountered, there should be a series of HTML files in the `docs/_build` directory that contain the project's documentation. To view them, simply open `docs/_build/index.html` with a web browser.
N.B.: If step 3 hits an error, it probably indicates that you already had Sphinx installed, and you're just a little unlucky today. Make sure your system is using the Python3 version of `Sphinx`. If it isn't, delete those Sphinx files or edit your `PYTHONPATH` to point to the Python3 Sphinx version. (On Debian-based Linux distributions, you can delete these files by executing `apt-get remove python2-sphinx`.

### requirements.txt vs. setup.py
I don't expect any users to need `requirements.txt` right now. Instead, the project should be installed using `pip install -e .`, which uses `setup.py`. I'm keeping `requirements.txt` around for now, though, in case it comes in handy later for some reason. Also, since `figure_generation/graph_creation.py` is essentially the entrypoint for the entire module and it uses absolute (`from figure_generation.figures import ...`) instead of relative (`from .figures import ...`) imports, it's necessary for install the `deepcell-kiosk-figure-generation` package (using `setup.py`), and installing package using only `requirements.txt` wouldn't be sufficient.
