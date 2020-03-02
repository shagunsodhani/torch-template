import torch

from codes.model.types import ModelType
from codes.utils import utils
from codes.utils.config import ConfigType


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
