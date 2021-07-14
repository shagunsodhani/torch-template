from copy import deepcopy
from typing import Dict

from src.dataset.types import DataLoaderType
from src.utils import utils
from src.utils.config import ConfigType


def build(config: ConfigType) -> Dict[str, DataLoaderType]:
    """Build the dataset

    Args:
        config (ConfigType): config to build the model

    Returns:
        DatasetType:
    """

    dataset_config = deepcopy(config.dataset)

    fn_to_build_dataloader = utils.load_callable(
        dataset_config["cls"] + ".get_dataloaders"
    )

    return fn_to_build_dataloader(config=config)  # type: ignore
