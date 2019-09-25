Recreate the figures from https://www.biorxiv.org/content/10.1101/505032v3 in 4 easy steps!

N.B.: Before following any of these steps, I'd recommend activating a Python `virtualenv`. It won't affect how easy it is to run this project, probably, but it will make your life easier after you've forgotten about this project and moved on to other Python packages.

## Figure Creation
1) To create the figures, first install the `deepcell-kiosk-figure-generation` package using the included `setup.py` file with the command `pip install -e .`.
2) Place all JSON benchmarking data in a top-level folder inside the project. (The project comes with a `new_data` folder containing default data for this purpose, but feel to throw this data away and replace it with your own, or put yours in an entirely new folder.)
3) Edit the global attributes in `figure_generation/graph_creation.py` to suit your tastes. (The global attributes should all be documented at the top of `figure_generation/graph_creation.py`.)
4) Execute `python3 ./figure_generation/graph_creation.py`.

Note that we did a significant amount of aesthetic primping in Illustrator to the figures generated from the included default data before we put them in the paper, but the data is all the same.

## Documentation
If you want to see the documentation for this project, follow these steps:
1) First, install the package with the `docs` dependencies included: `pip install -e .[docs]`. (It's ok if you've already installed without the docs dependencies; there won't be any conflict if you use this command after that.)
2) Move into the `docs` directory.
3) Execute `sphinx-build -b html . ./_build/`. Assuming no errors were encountered, there should be a series of HTML files in the `docs/_build` directory that contain the project's documentation. To view them, simply open `docs/_build/index.html` with a web browser.
N.B.: If step 3 hits an error, it probably indicates that you already had Sphinx installed, and you're just a little unlucky today. Make sure your system is using the Python3 version of `Sphinx`. If it isn't, delete those Sphinx files or edit your `PYTHONPATH` to point to the Python3 Sphinx version. (On Debian-based Linux distributions, you can delete these files by executing `apt-get remove python2-sphinx`.
