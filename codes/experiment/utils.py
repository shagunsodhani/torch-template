from __future__ import print_function

import shutil
from typing import List

from codes.experiment.experiment import Experiment
from codes.utils.config import ConfigType


def prepare_and_run(config: ConfigType) -> None:
    """Prepare an experiment and run the experiment

    Args:
        config (ConfigType): config of the experiment
    """
    experiment = Experiment(config)
    experiment.run()


def clear(config: ConfigType) -> None:
    """Clear an experiment and delete all its data/metadata/logs
    given a config

    Args:
        config (ConfigType): config of the experiment to be cleared
    """

    for dir_to_del in get_dirs_to_delete_from_experiment(config):
        shutil.rmtree(dir_to_del)


def get_dirs_to_delete_from_experiment(config: ConfigType) -> List[str]:
    """Return a list of dirs that should be deleted when clearing an
        experiment

    Args:
        config (ConfigType): config of the experiment to be cleared

    Returns:
        List[str]: List of directories to be deleted
    """
    return [config.logbook.dir, config.experiment.save_dir]
