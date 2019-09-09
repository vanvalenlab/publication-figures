# utils.py

# custom execption for a failing regex
class NoGpuError(Exception):
    pass

# custom execption for missing data
class MissingDataError(Exception):
    pass

# https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
def pretty(json_input, indent=0):
    for key, value in json_input.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
