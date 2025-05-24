# use logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logger to output to standard output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# from stackoverflow.com https://stackoverflow.com/questions/6086976/how-to-get-a-complete-exception-stack-trace-in-python
def full_stack():
    import traceback, sys

    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]  # remove call of full_stack, the printed exception
        # will contain the caught exception caller instead
    trc = "Traceback (most recent call last):\n"
    stackstr = trc + "".join(traceback.format_list(stack))
    if exc is not None:
        stackstr += "  " + traceback.format_exc().lstrip(trc)
    return stackstr


import os
from collections import defaultdict


def get_unique_short_names(paths):
    # Normalize paths and split into parts
    path_parts = [os.path.normpath(p).split(os.sep) for p in paths]

    # Start with just the basename
    name_map = defaultdict(list)
    for i, parts in enumerate(path_parts):
        base = parts[-1]
        name_map[base].append((i, parts))

    result = [None] * len(paths)

    for base, items in name_map.items():
        if len(items) == 1:
            # No conflict
            i, _ = items[0]
            result[i] = base
        else:
            # Resolve conflict by prepending dirs
            max_depth = max(len(p) for _, p in items)
            for depth in range(2, max_depth + 1):
                temp_names = {}
                conflict = False
                for i, parts in items:
                    short = os.path.join(*parts[-depth:])
                    if short in temp_names:
                        conflict = True
                        break
                    temp_names[short] = i
                if not conflict:
                    # All names are unique with current depth
                    for short, i in temp_names.items():
                        result[i] = short
                    break
            else:
                # Fallback to full path if nothing else works
                for i, parts in items:
                    result[i] = os.path.join(*parts)

    return result


def interpret_file_relative_to(base_dir, fpath):
    full_path = base_dir / fpath
    print(f"full_path: {full_path}")
    if not full_path.exists():
        logger.warning(f"File {full_path} does not exist. Using {fpath} instead.")
        full_path = fpath
    print(f"full_path: {full_path}")
    return full_path
