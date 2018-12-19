"""Code to bootstrap the metrics"""
from copy import deepcopy

from codes.utils.util import merge_nested_dicts


def get_default_metric_dict(mode):
    """Method to return a defualt metric dict"""
    return dict(
        mode=mode,
        num_timesteps=0,
        fps=0,
        episodic_rewards=0,
        time=0,
        num_update=-1,
    )


def merge_metric_dicts(current_metric_dict, new_metric_dict):
    """Method to merge multiple metric dicts into one dict"""
    return merge_nested_dicts(current_metric_dict, new_metric_dict)


def prepare_metric_dict_to_log(current_metric_dict, num_updates):
    """Method to prepare the metric dict before writing to the logger"""
    metric_dict = deepcopy(current_metric_dict)
    metric_dict["num_updates"] = num_updates
    return metric_dict
