"""Code to read and process the config"""
import datetime
import os

import torch
import yaml

from codes.utils.log import set_logger, write_message_logs, write_config_log
from codes.utils.serializable_config import get_config_box, get_forzen_config_box
from codes.utils.util import make_dir, \
    get_current_commit_id, merge_nested_dicts


def read_config_file(config_id="config"):
    """Method to read a config file"""
    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    config_name = "{}.yaml".format(config_id)
    return yaml.load(open(os.path.join(path, 'config', config_name)))


def get_config(config_id=None, should_make_dir=True, experiment_id=0):
    """Method to prepare the config for all downstream tasks"""

    sample_config = read_config_file("sample_config")
    actual_config = read_config_file(config_id)
    merged_config = merge_nested_dicts(sample_config, actual_config)
    boxed_config = get_config_box(merged_config)
    config = _post_process(boxed_config, should_make_dir, experiment_id)
    if _is_valid_config(config, config_id):
        return config
    return None


def _is_valid_config(config, config_id):
    """Simple tests to check the validity of a given config file"""
    if config.general.id == config_id:
        return True
    print("Error in Config. Config Id and Config Names do not match")
    return False


def _post_process(config, should_make_dir, experiment_id=0):
    """Post Processing on the config"""

    config.general = _post_process_general_config(config.general, experiment_id)
    config.model = _post_process_model_config(config.model, config, should_make_dir)
    config.log = _post_process_log_config(config.log, config.general, should_make_dir)
    config.plot = _post_process_plot_config(config.plot, config.general, should_make_dir)
    return get_forzen_config_box(config.to_dict())


def _post_process_general_config(general_config, experiment_id=0):
    """Method to post process the general section of the config"""

    if not general_config.base_path:
        general_config.base_path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]

    if not general_config.commit_id:
        general_config.commit_id = get_current_commit_id()

    if not general_config.date:
        general_config.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    general_config.device = torch.device(general_config.device)  # pylint: disable=E1101
    general_config.experiment_id = experiment_id
    env_var_name = "SLURM_JOB_ID"
    if env_var_name in os.environ:
        general_config.slurm_id = str(os.environ[env_var_name])

    return general_config


def _post_process_model_config(model_config, config, should_make_dir):
    """Method to post process the model section of the config"""

    general_config = config.general

    if not model_config.save_dir:
        model_config.save_dir = os.path.join(general_config.base_path, "model", general_config.id)
    elif model_config.save_dir[0] != "/":
        model_config.save_dir = os.path.join(general_config.base_path, model_config.save_dir)

    if should_make_dir:
        make_dir(path=model_config.save_dir)

    model_config.load_path = os.path.join(general_config.base_path,
                                          "model", model_config.load_path)

    return model_config


def _post_process_plot_config(plot_config, general_config, should_make_dir):
    """Method to post process the plot section of the config"""
    if not plot_config.base_path:
        plot_config.base_path = os.path.join(general_config.base_path,
                                             "plots", general_config.id)
        if should_make_dir:
            make_dir(path=plot_config.base_path)

    return plot_config


def _post_process_log_config(log_config, general_config, should_make_dir):
    """Method to post process the log section of the config"""

    if not log_config.file_path:
        log_config.file_path = os.path.join(general_config.base_path,
                                            "logs", general_config.id)
        if should_make_dir:
            make_dir(path=log_config.file_path)
            make_dir(os.path.join(log_config.file_path, "train"))
            make_dir(os.path.join(log_config.file_path, "eval"))
        log_config.file_path = os.path.join(log_config.file_path, "log.txt")

    log_config.dir = log_config.file_path.rsplit("/", 1)[0]
    return log_config


if __name__ == "__main__":
    config = get_config("sample_config")
    set_logger(config)
    write_message_logs("torch version = {}".format(torch.__version__))
    write_config_log(config)
