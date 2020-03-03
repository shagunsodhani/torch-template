import time

import torch

from codes.utils.config import ConfigType, get_config
from codes.utils.utils import set_seed, write_debug_message


def bootstrap_config(
    config_id: str, should_make_config_immutable: bool = True
) -> ConfigType:
    """Prepare the config object

    Args:
        config_id (str): config_id to load
        should_make_config_immutable (bool, optional): Should the config object
            be immutable. Defaults to True.

    Returns:
        ConfigType: Config Object
    """
    config = get_config(
        config_id, should_make_config_immutable=should_make_config_immutable
    )
    write_debug_message(
        f"Starting Experiment at {time.asctime(time.localtime(time.time()))}"
    )
    write_debug_message(f"torch version = {torch.__version__}")  # type: ignore
    set_seed(seed=config.general.seed)
    return config
