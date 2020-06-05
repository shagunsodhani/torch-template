"""Base class for all the models (with batteries)"""
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

import hydra
from codes.model.types import OptimizerType, SchedulerType
from codes.model.utils import OptimizerSchedulerTuple
from codes.utils import utils
from codes.utils.checkpointable import Checkpointable
from codes.utils.config import ConfigType


class BaseModel(nn.Module, Checkpointable):  # type: ignore
    """Base class for all models"""

    def __init__(self, config: ConfigType, device: torch.device):
        super().__init__()
        self.config = config
        self.name = "base_model"
        self.description = (
            "This is the base class for all the models. "
            "All the other models should extend this class. "
            "It is not to be used directly."
        )
        self.epsilon = 1e-6
        self.device = device

    def save(  # type: ignore
        self,
        epoch: int,
        optimizer_scheduler_tuple: OptimizerSchedulerTuple,
        is_best_model: bool = False,
        path_to_save_to: str = "",
    ) -> str:
        """Method to persist the model.
        Note this method is not well tested"""
        utils.make_dir(path_to_save_to)
        # Updating the information about the epoch
        # Check if the epoch_state is already saved on the file system
        epoch_state_path = os.path.join(path_to_save_to, "epoch.tar")

        if os.path.exists(epoch_state_path):
            epoch_state = torch.load(epoch_state_path)  # type: ignore
        else:
            epoch_state = {}
        epoch_state["current"] = epoch
        if is_best_model:
            epoch_state["best"] = epoch
        torch.save(epoch_state, epoch_state_path)  # type: ignore

        state = {
            "metadata": {"epoch": epoch, "is_best_model": is_best_model},
            "model": {"state_dict": self.state_dict()},
            "optimizer": {
                "state_dict": optimizer_scheduler_tuple.optimizer.state_dict()
            },
            "random_state": {
                "np": np.random.get_state(),
                "python": random.getstate(),
                "pytorch": torch.get_rng_state(),  # type: ignore
            },
        }
        scheduler = optimizer_scheduler_tuple.scheduler

        if scheduler:
            state["scheduler"] = {"state_dict": scheduler.state_dict()}

        path = self._get_path_to_save_model(
            dir_to_save_experiment=path_to_save_to, epoch=epoch
        )

        torch.save(state, path)  # type: ignore
        return "Saved experiment to path = {}".format(path)

    def _get_path_to_save_model(self, dir_to_save_experiment: str, epoch: int) -> str:
        return os.path.join(dir_to_save_experiment, "epoch_{}.tar".format(epoch))

    def load(  # type: ignore
        self,
        epoch: int,
        optimizer_scheduler_tuple: OptimizerSchedulerTuple,
        path_to_load_from: str,
        should_load_optimizer: bool = True,
    ) -> Tuple[OptimizerSchedulerTuple, str]:
        """Public method to load the model"""

        checkpoint = torch.load(path_to_load_from)  # type: ignore
        load_random_state(checkpoint["random_state"])
        self._load_model_params(checkpoint["model"]["state_dict"])

        if should_load_optimizer:
            optimizer = optimizer_scheduler_tuple.optimizer
            optimizer.load_state_dict(checkpoint["optimizer"]["state_dict"])

            key = "scheduler"
            scheduler = None
            if key in checkpoint:
                scheduler = optimizer_scheduler_tuple.scheduler
                if scheduler is not None:
                    scheduler.load_state_dict(checkpoint[key]["state_dict"])
        message = "Loading model from path {}".format(path_to_load_from)

        return (
            OptimizerSchedulerTuple(optimizer=optimizer, scheduler=scheduler),
            message,
        )

    def _load_model_params(self, state_dict: Dict[str, Any]) -> Any:
        """Method to load the model params"""
        self.load_state_dict(state_dict)

    def get_model_params(self) -> List[torch.nn.Parameter]:
        """Method to get the model params"""
        model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        # num_params = sum([torch.numel(p) for p in model_parameters])
        # write_message("Total number of params = " + str(params))
        return model_parameters

    def get_optimizer_and_scheduler(self) -> OptimizerSchedulerTuple:
        """Return a OptimizerSchedulerTuple for the model"""
        optimizer = self.get_optimizer()
        scheduler = self._register_optimizer_to_scheduler(optimizer)
        return OptimizerSchedulerTuple(optimizer=optimizer, scheduler=scheduler)

    def get_optimizer(self) -> OptimizerType:
        """Return an optimizer for the model"""
        model_params = self.get_model_params()
        return self._register_params_to_optimizer(model_params)

    def _register_params_to_optimizer(
        self, model_params: List[torch.nn.Parameter]
    ) -> OptimizerType:
        optimizer = hydra.utils.instantiate(self.config.optim, model_params)
        assert isinstance(optimizer, OptimizerType)
        return optimizer

    def _register_optimizer_to_scheduler(
        self, optimizer: OptimizerType
    ) -> Optional[SchedulerType]:

        scheduler_config = deepcopy(self.config.scheduler)
        if scheduler_config:
            scheduler = hydra.utils.instantiate(self.config.scheduler)
            assert isinstance(scheduler, SchedulerType)

            return scheduler

        return None

    def forward(self, data):  # type: ignore
        # pylint: disable=W0221,W0613
        """Forward pass of the network"""
        raise NotImplementedError

    def get_param_count(self) -> int:
        """Count the number of params"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum((np.prod(p.size()) for p in model_parameters))

    def freeze_weights(self) -> None:
        """Freeze the model"""
        for param in self.parameters():
            param.requires_grad = False

    def __str__(self) -> str:
        """Return string description of the model"""
        return self.description


def load_random_state(random_state: Dict[str, Any]) -> None:
    """Method to load the random state"""
    np.random.set_state(random_state["np"])
    random.setstate(random_state["python"])
    torch.set_rng_state(random_state["pytorch"])  # type: ignore
