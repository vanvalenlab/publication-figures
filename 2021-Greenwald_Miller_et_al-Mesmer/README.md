This repo contains the code to reproduce the figures from Greenwald, Miller et al.

The data to generate the figures can be found [here](https://storage.googleapis.com/publications-data/mesmer-preprint/mesmer_publication_data.zip): please download it and place it in the top-level directory for the example code to run correctly. 

All of the scripts to directly generate the figures are provided in the top-level directory, numbered with the figure they correspond to. Helper functions are present in the *figures.py* file. 

The notebooks used for model training and evaluation can be found in the *notebooks* folder. The corresponding data for model training can be found [here](tbd.com). 
These notebooks are specifically for the benchmarking that was performed in the paper. 

For a general template of how to train the Mesmer model, see the *Mesmer_training_notebook.ipynb* file in the top level directory. 

For information on how to use Mesmer to analyze your data, see our [introductory repo](https://github.com/vanvalenlab/intro-to-deepcell/tree/master/pretrained_models)
