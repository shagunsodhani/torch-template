"""Collection of utility functions"""

# pylint: disable=C0103

import gc
import importlib
import json
import os
import pathlib
import random
import re
import subprocess
from functools import reduce, wraps
from operator import mul
from time import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

T = TypeVar("T")


def write_debug_message(message: str) -> None:
    """Use the logbook instead of using this method.

    Method to write the debug logs to the console. This is only meant to
    be used when the logbook is not constructued. In all other cases, use
    the logbook.

    Args:
        message (str): message to write
    """
    log = {
        "message": message,
        "logbook_type": "debug",
    }
    print(json.dumps(log))


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "#"
) -> Dict[str, Any]:
    """Method to flatten a given dict using the given seperator.
    Taken from https://stackoverflow.com/a/6027615/1353861
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(dictionary: Dict[str, Any], sep: str = "#") -> Dict[str, Any]:
    """Method to flatten a given dict using the given seperator.
        Taken from https://stackoverflow.com/questions/6037503/python-unflatten-dict
        """
    resultDict: Dict[str, Any] = {}
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def grouped(iterable: Iterable[T], num_elements_to_group: int) -> Iterable[Iterable[T]]:
    """Modified from
    https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list/39038787"""
    return zip(*[iter(iterable)] * num_elements_to_group)


def padarray(A: np.array, size: int, const: float = 1.0) -> np.array:
    """Taken from
    https://stackoverflow.com/questions/38191855/zero-pad-numpy-array/38192105"""
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode="constant", constant_values=const)


def shuffle_list(*ls: Iterable[T]) -> Iterable[T]:
    """Taken from
    https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order"""
    input_list = list(zip(*ls))
    random.shuffle(input_list)
    return zip(*input_list)


def chunks(l: List[T], n: int) -> Iterator[List[T]]:
    """
    Taken from
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Yield successive n-sized chunks from l."""
    for idx in range(0, len(l), n):
        yield l[idx : idx + n]  # noqa: E203


def reverse_dict(_dict: Dict[Any, Any]) -> Dict[Any, Any]:
    """Reverse Dict"""
    return {v: k for k, v in _dict.items()}


def make_dir(path: str) -> None:
    """Make dir"""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


# def get_device_name(device_type):
#     """Get device name"""
#     if torch.cuda.is_available() and "cuda" in device_type:
#         return device_type
#     return "cpu"


def get_current_commit_id() -> str:
    """Get current commit id

    Returns:
        str: Current commit id
    """
    command = "git rev-parse HEAD"
    commit_id = subprocess.check_output(command.split()).strip().decode("utf-8")
    return commit_id


def set_seed(seed: int) -> None:
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # type: ignore
        # Module has no attribute "manual_seed_all"  [attr-defined]
    os.environ["PYTHONHASHSEED"] = str(seed)


def timing(f: Callable[..., Any]) -> Callable[..., Any]:
    """Timing function"""

    @wraps(f)
    def wrap(*args: Any, **kw: Any) -> Any:
        ts = time()
        result = f(*args, **kw)
        te = time()
        write_debug_message("function:{} took: {} sec".format(f.__name__, te - ts))
        return result

    return wrap


def show_tensor_as_image(_tensor: np.asarray) -> None:
    """Plot a tensor as image"""
    plt.imshow(_tensor.astype(np.uint8), origin="lower")
    plt.show()


def save_tensor_as_image(_tensor: np.asarray, file_path: str) -> None:
    """Save a tensor as image """
    plt.imsave(file_path, _tensor.astype(np.uint8), origin="lower")


def get_product_of_iterable(
    _iterable: Iterable[Union[float, int]]
) -> Union[float, int]:
    """Method to get the product of all the enteries in an iterable"""
    return reduce(mul, _iterable, 1)


def log_pdf(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Method to compute the log pdf for x under a gaussian
    distribution with mean = mu and standard deviation = std
    Taken from: https://chrisorm.github.io/VI-MC-PYT.html"""

    return -0.5 * torch.log(2 * np.pi * std ** 2) - (
        0.5 * (1.0 / (std ** 2)) * (x - mu) ** 2  # type: ignore
    )  # pylint: disable=no-member
    # Unsupported operand types for / ("float" and "Tensor")  [operator]


def running_mean(x: np.array, N: int) -> np.array:
    """Taken from
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean"""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def merge_first_two_dims(batch: torch.Tensor) -> torch.Tensor:
    """Merge first two dims in a batch"""
    shape = batch.shape
    return batch.view(-1, *shape[2:])


def unmerge_first_and_second_dim(
    batch: torch.Tensor, first_dim: int = -1, second_dim: int = -1
) -> torch.Tensor:
    """Method to modify the shape of a batch by unmerging the first dimension.
    Given a tensor of shape (a*b, c, ...), return a tensor of shape (a, b, c, ...)"""
    shape = batch.shape
    return batch.view(first_dim, second_dim, *shape[1:])


def merge_second_and_third_dim(batch: torch.Tensor) -> torch.Tensor:
    """Merge the second and the third dims in a batch.
    Used when flattening messages from the primitives ot the master"""
    shape = batch.shape
    return batch.view(shape[0], int(shape[1] * shape[2]), *shape[3:])


def unmerge_second_and_third_dim(
    batch: torch.Tensor, second_dim: int = -1, third_dim: int = -1
) -> torch.Tensor:
    """Method to modify the shape of a batch by unmerging the second and the third dimension.
    Given a tensor of shape (a, b*c, ...), return a tensor of shape (a, b, c, ...)"""
    shape = batch.shape
    return batch.view(second_dim, third_dim, *shape[1:])


def merge_nested_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Method to merge dict1 and dict2.
        dict1 is over written
        """
    sep = "#"
    flattened_dict1 = flatten_dict(dict1, sep=sep)
    flattened_dict2 = flatten_dict(dict2, sep=sep)
    flattened_merged_dict = {**flattened_dict1, **flattened_dict2}
    return unflatten_dict(flattened_merged_dict, sep=sep)


# def map_observation_space_to_shape(obs):
#     """Method to obtain the shape from observation space"""

#     if obs.__class__.__name__ == "Discrete":
#         return (obs.n,)
#     if obs.__class__.__name__ == "Box":
#         return obs.shape
#     return obs.shape


def split_on_caps(input_str: str) -> List[str]:
    """Method to split a given string at uppercase characters.
    Taken from: https://stackoverflow.com/questions/2277352/split-a-string-at-uppercase-letters
    """
    return re.findall("[A-Z][^A-Z]*", input_str)


def print_mem_report() -> None:
    """Method to print the memory usage by different tensors.
    Taken from the pytorch forums."""
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            write_debug_message(f"{type(obj)}, {obj.size()}")


def get_cpu_stats() -> Dict[str, str]:
    """Method to return/print the CPU stats. Taken from pytorch forums """
    cpu_percent = psutil.cpu_percent()
    virtual_memory = psutil.virtual_memory()
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / (2.0 ** 30)  # memory use in GB...I think
    # print('memory GB:', memoryUse)
    return {
        "cpu_percent": cpu_percent,
        "virtual_memory": virtual_memory,
        "memory_use": memory_use,
    }


def load_callable(callable_name: str) -> Any:
    """Load a callable object - function, class, etc

    Args:
        callable_name (str): Name of the callable

    Returns:
        Any: Loaded callable
    """
    module_name, attr_name = callable_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr_name)
