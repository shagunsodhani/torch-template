"""Code to interface with the config."""
import datetime
import os
from typing import Any, Dict, cast

from omegaconf import OmegaConf

from codes.utils import utils
from codes.utils.types import ConfigType

# ConfigType = Union[DictConfig]


def make_config_mutable(config: ConfigType) -> ConfigType:
    """Set the config to be mutable.

    Args:
        config (ConfigType):

    Returns:
        ConfigType:
    """
    OmegaConf.set_readonly(config, False)
    return config


def make_config_immutable(config: ConfigType) -> ConfigType:
    """Set the config to be immutable.

    Args:
        config (ConfigType):

    Returns:
        ConfigType:
    """
    OmegaConf.set_readonly(config, True)
    return config


def to_dict(config: ConfigType) -> Dict[str, Any]:
    """Convert config to serializable dictionary.

    Args:
        config (ConfigType):

    Returns:
        Dict:
    """
    dict_config = cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))
    return dict_config


def process_config(config: ConfigType, should_make_dir: bool = True) -> ConfigType:
    """Process the config.

    Args:
        config (ConfigType): Config object
        should_make_dir (bool, optional): Should make dir for saving logs, models etc. Defaults to True.

    Returns:
        ConfigType: Processed config
    """
    config = OmegaConf.create(to_dict(config))  # resolving the config

    config = _process_general_config(config=config)
    config = _process_experiment_config(config=config, should_make_dir=should_make_dir)
    return config


def _process_general_config(config: ConfigType) -> ConfigType:
    """Process the `general` section of the config

    Args:
        config (ConfigType): Config object

    Returns:
        [ConfigType]: Processed config
    """

    general_config = config.general

    if not general_config.commit_id:
        general_config.commit_id = utils.get_current_commit_id()

    if not general_config.date:
        general_config.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    slurm_id = []
    env_var_names = ["SLURM_JOB_ID", "SLURM_STEP_ID"]
    for var_name in env_var_names:
        if var_name in os.environ:
            slurm_id.append(str(os.environ[var_name]))
    if slurm_id:
        general_config.slurm_id = "-".join(slurm_id)

    return config


def _process_experiment_config(config: ConfigType, should_make_dir: bool) -> ConfigType:
    """Process the `experiment` section of the config

    Args:
        config (ConfigType): Config object
        should_make_dir (bool): Should make dir for the data

    Returns:
        ConfigType: Processed config
    """
    if should_make_dir:
        utils.make_dir(path=config.experiment.save_dir)
    return config
