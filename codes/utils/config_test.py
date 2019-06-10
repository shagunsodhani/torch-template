"""Code to test the config parser"""

import torch

from codes.utils.config import get_config
from codes.utils.log import set_logger, write_message_logs, write_config_log


def test_parser():
    """Method to test if the config parser can load the config file correctly"""
    config_name = "sample_config"
    config = get_config(config_name)
    set_logger(config)
    write_message_logs("torch version = {}".format(torch.__version__))
    assert config.general.id == config_name


def test_serialization():
    """Method to test if the config object is serializable"""
    config_name = "sample_config"
    config = get_config(config_name)
    set_logger(config)
    assert write_config_log(config) is None


if __name__ == "__main__":
    test_parser()
    test_serialization()
