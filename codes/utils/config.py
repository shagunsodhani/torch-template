"""Code to interface with the config"""
import datetime
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union, cast

from omegaconf import DictConfig, OmegaConf

from codes.utils import utils

ConfigType = Union[DictConfig]


def make_config_mutable(config: ConfigType) -> ConfigType:
    """Set the config to be mutable

    Args:
        config (ConfigType):

    Returns:
        ConfigType:
    """
    OmegaConf.set_readonly(config, False)
    return config


def make_config_immutable(config: ConfigType) -> ConfigType:
    """Set the config to be immutable

    Args:
        config (ConfigType):

    Returns:
        ConfigType:
    """
    OmegaConf.set_readonly(config, True)
    return config


def to_dict(config: ConfigType) -> Dict[str, Any]:
    """Convert config to serializable dictionary

    Args:
        config (ConfigType):

    Returns:
        Dict:
    """
    dict_config = cast(Dict[str, Any], OmegaConf.to_container(config))
    return dict_config


def _read_config_file_and_load_components(config_id: str = "config") -> ConfigType:
    """Read a config file

    Args:
        config_id (str, optional): Id of the config file to read.
            Defaults to "config".

    Returns:
        ConfigType: Config object
    """

    config = read_config_file(config_id=config_id)
    for key in config:
        config[key] = _load_components(config[key])
    return config


def read_config_file(config_id: str = "sample_config") -> ConfigType:
    """Read a config file

    Args:
        config_id (str, optional): Id of the config file to read.
            Defaults to "sample_config".

    Returns:
        ConfigType: Config object
    """

    project_root = str(Path(__file__).resolve()).split("/codes")[0]
    config_name = "{}.yaml".format(config_id)
    config = OmegaConf.load(os.path.join(project_root, "config", config_name))
    assert isinstance(config, DictConfig)
    return config


def get_config(
    config_id: str,
    should_make_dir: bool = True,
    should_make_config_immutable: bool = True,
) -> ConfigType:
    """Prepare the config for all downstream tasks

    Args:
        config_id (str): Id of the config file to read.
        should_make_dir (bool, optional): Should make dir (for saving
            models, logs etc). Defaults to True.
        should_make_config_immutable (bool, optional): Should the config be frozen (immutable).
             Defaults to True.

    Returns:
        ConfigType: [description]
    """
    sample_config = _read_config_file_and_load_components("sample_config")
    current_config = _read_config_file_and_load_components(config_id)
    config = OmegaConf.merge(sample_config, current_config)
    OmegaConf.set_struct(config, True)
    resolved_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    assert isinstance(resolved_config, DictConfig)
    config = _process(config=resolved_config, should_make_dir=should_make_dir)
    if should_make_config_immutable:
        config = make_config_immutable(config)
    assert _is_valid_config(config, config_id)
    return config


def _load_components(config: ConfigType) -> ConfigType:
    """Load the different componenets in a config

    Args:
        config (ConfigType)

    Returns:
        ConfigType
    """
    special_key = "_load"
    if config is not None and special_key in config:
        loaded_config = read_config_file(config.pop(special_key))
        updated_config = OmegaConf.merge(loaded_config, config)
        assert isinstance(updated_config, ConfigType)
        return updated_config
    return config


def _is_valid_config(config: ConfigType, config_id: str) -> bool:
    """Check if a config is valid

    Args:
        config (ConfigType): Config object
        config_id (str): Config id to verify the config object

    Returns:
        bool: Is the config object valid
    """
    if config.general.id == config_id.replace("/", "_"):
        return True

    utils.write_debug_message(
        message="Error in Config. Config Id and Config Names do not match"
    )
    return False


def _process(config: ConfigType, should_make_dir: bool) -> ConfigType:
    """Process the config

    Args:
        config (ConfigType): Config object
        should_make_dir (bool): Should make dir for saving logs, models etc

    Returns:
        [ConfigType]: Processed config
    """

    config = _process_general_config(config=config)
    config = _process_logbook_config(config=config, should_make_dir=should_make_dir)
    config = _process_experiment_config(config=config, should_make_dir=should_make_dir)
    return config


def _process_general_config(config: ConfigType) -> ConfigType:
    """Process the `general` section of the config

    Args:
        config (ConfigType): Config object

    Returns:
        [ConfigType]: Processed config
    """

    general_config = deepcopy(config.general)
    general_config.id = general_config.id.replace("/", "_")

    if not general_config.commit_id:
        general_config.commit_id = utils.get_current_commit_id()

    if not general_config.date:
        general_config.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    slurm_id = []
    env_var_names = ["SLURM_JOB_ID", "SLURM_STEP_ID"]
    for var_name in env_var_names:
        if var_name in os.environ:
            slurm_id.append(str(os.environ[var_name]))
    if slurm_id:
        general_config.slurm_id = "-".join(slurm_id)

    config.general = general_config

    return config


def _process_experiment_config(config: ConfigType, should_make_dir: bool) -> ConfigType:
    """Process the `experiment` section of the config

    Args:
        config (ConfigType): Config object
        should_make_dir (bool): Should make dir for the data

    Returns:
        ConfigType: Processed config
    """
    experiment_config = config.experiment
    if should_make_dir:
        utils.make_dir(path=experiment_config.save_dir)
    return config


def _process_logbook_config(config: ConfigType, should_make_dir: bool) -> ConfigType:
    """Process the `logbook` section of the config

    Args:
        config (ConfigType): Config object
        should_make_dir (bool): Should make a dir to save the logs

    Returns:
        ConfigType: Processed config
    """
    logbook_config = config.logbook
    if should_make_dir:
        utils.make_dir(path=logbook_config.dir)
        utils.make_dir(path=logbook_config.tensorboard.logdir)

    return config
