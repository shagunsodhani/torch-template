import torch

from src.model.types import ModelType
from src.utils import utils
from src.utils.config import ConfigType


def build(config: ConfigType, device: torch.device) -> ModelType:
    """Build the model

    Args:
        config (ConfigType): config to build the model
        device (torch.device): device to put the model on

    Returns:
        ModelType
    """

    model_config = config.model

    model_cls = utils.load_callable(model_config["cls"])

    return model_cls(config=config, device=device).to(device)  # type: ignore
