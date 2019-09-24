# utils.py
""" Odds and ends that might be useful in other files."""


# To handle relative imports gracefully
if __name__ == "__main__" and __package__ is None:
    __package__ = "figure_generation"


# custom exception for a failing regex
class NoGpuError(Exception):
    """ This error is raised when a run is done without GPUs (meaning it uses only CPUs for
        Tensorflow predictions, which is, in theory, possible).

    """

# custom exception for missing data
class MissingDataError(Exception):
    """ This exception is raised when some ...Figure class tries to pare down self.raw_data in such
        a way that it removes all data and is left with nothing that meets its criteria. It
        indicates that we're trying to graph data that we don't have.

    """

# https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(json_input, indent=0):
    for key, value in json_input.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
