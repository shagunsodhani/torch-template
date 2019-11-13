"""Base class for all the models (with batteries)"""
import importlib
import os
import random

from typing import List, Optional
import numpy as np
import torch
from torch import nn, optim

from codes.logbook.filesystem_logger import write_message_logs
from codes.utils.checkpointable import Checkpointable


class BaseModel(nn.Module, Checkpointable):
    """Base class for all models"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = "base_model"
        self.description = "This is the base class for all the models. " \
                           "All the other models should extend this class. " \
                           "It is not to be used directly."
        self.criteria = nn.MSELoss(reduction="mean")
        self.epsilon = 1e-6

    def loss(self, outputs, labels):
        """Method to compute the loss"""
        return self.criteria(outputs, labels)

    def track_loss(self, outputs, labels):
        """There are two different functions related to loss as we might be interested in
         tracking one loss and optimising another"""
        return self.loss(outputs, labels)

    def save(
        self,
        epoch: int,
        optimizers: Optional[List[torch.optim.Optimizer]],
        is_best_model: bool = False,
    ) -> None:
        """Method to persist the model.
        Note this method is not well tested"""
        model_config = self.config.model
        # Updating the information about the epoch
        # Check if the epoch_state is already saved on the file system
        epoch_state_path = os.path.join(model_config.save_dir, "epoch.tar")

        if os.path.exists(epoch_state_path):
            epoch_state = torch.load(epoch_state_path)
        else:
            epoch_state = {"best": epoch}
        epoch_state["current"] = epoch
        if is_best_model:
            epoch_state["best"] = epoch
        torch.save(epoch_state, epoch_state_path)

        state = {
            "metadata": {"epoch": epoch, "is_best_model": False,},
            "model": {"state_dict": self.state_dict(),},
        "optimizers": [{"state_dict": optimizer.state_dict()}
                for optimizer in self.get_optimizers()],
            "random_state": {
                "np": np.random.get_state(),
                "python": random.getstate(),
                "pytorch": torch.get_rng_state(),
            },
            # "schedulers": [scheduler.state_dict() for scheduler in schedulers]
        }
        
        path = os.path.join(
            model_config.save_dir, "experiment_epoch_{}.tar".format(epoch)
        )

        if is_best_model:
            state["metadata"]["is_best_model"] = True
        else:
        torch.save(state, path)
        write_message_logs("saved experiment to path = {}".format(path))

    def load(
        self,
        epoch: int,
        should_load_optimizers: bool = True,
        optimizers=Optional[List[optim.Optimizer]],
        schedulers=Optional[List[optim.lr_scheduler.ReduceLROnPlateau]],
    ) -> None:
        """Public method to load the model"""
        

        model_config = self.config.model
        load_path = model_config.load_path
        if load_path == "":
            # We are resuming an experiment
            load_path = "{}/experiment_epoch_{}.tar".format(
                self.config.general.id, epoch
            )
        elif load_path[-1] == "/":
            load_path = load_path[:-1]
        path = "{}/{}_agent_id_{}.tar".format(load_path,
                                              self.config.general.experiment_id,
                                              index)
        if not os.path.exists(path):
            path = "{}/{}_agent_id_{}.tar".format(load_path,
                                                  self.config.general.experiment_id,
                                                  0)
        write_message_logs("Loading model from path {}".format(path))
        if str(self.config.general.device) == "cuda":
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        load_random_state(checkpoint["random_state"])
        self._load_model_params(checkpoint["model"]["state_dict"])

        if should_load_optimizers:
            if optimizers is None:
                optimizers = self.get_optimizers()
            for optim_index, optimizer in enumerate(optimizers):
                optimizer.load_state_dict(
                    checkpoint["optimizers"][optim_index]["state_dict"]
                )
            key = "schedulers"
            if key in checkpoint:
                for scheduler_index, scheduler in enumerate(schedulers):
                    scheduler.load_state_dict(
                        checkpoint[key][scheduler_index]["state_dict"]
                    )
        return optimizers, schedulers

    def _load_model_params(self, state_dict):
        """Method to load the model params"""
        self.load_state_dict(state_dict)

    def get_model_params(self):
        """Method to get the model params"""
        model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        write_message_logs("Total number of params = " + str(params))
        return model_parameters

    def get_optimizers_and_schedulers(self):
        """Method to return the list of optimizers and schedulers for the model"""
        optimizers = self.get_optimizers()
        if optimizers:
            optimizers, schedulers = self._register_optimizers_to_schedulers(optimizers)
            return optimizers, schedulers
        return None

    def get_optimizers(self):
        """Method to return the list of optimizers for the model"""
        optimizers = []
        model_params = self.get_model_params()
        if model_params:
            optimizers.append(self._register_params_to_optimizer(model_params))
            return optimizers
        return None

    def _register_params_to_optimizer(self, model_params):
        """Method to map params to an optimizer"""
        optim_config = self.config.model.optim
        optimizer_cls = getattr(importlib.import_module("torch.optim"), optim_config.name)
        optim_name = optim_config.name.lower()
        if optim_name == "adam":
            return optimizer_cls(
                model_params,
                lr=optim_config.learning_rate,
                weight_decay=optim_config.weight_decay,
                eps=optim_config.eps
            )
        return optimizer_cls(
            model_params,
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            eps=optim_config.eps
        )

    def _register_optimizers_to_schedulers(self, optimizers):
        """Method to map optimzers to schedulers"""
        optimizer_config = self.config.model.optimizer
        if optimizer_config.scheduler_type == "exp":
            schedulers = list(map(lambda optimizer: optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=self.config.model.optimizer.scheduler_gamma),
                                  optimizers))
        elif optimizer_config.scheduler_type == "plateau":
            schedulers = list(map(lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                patience=self.config.model.optimizer.scheduler_patience,
                factor=self.config.model.optimizer.scheduler_gamma,
                verbose=True),
                                  optimizers))

        return optimizers, schedulers

    def forward(self, data):  # pylint: disable=W0221,W0613
        '''Forward pass of the network'''
        return None

    def get_param_count(self):
        """Count the number of params"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def freeze_weights(self):
        """Freeze the model"""
        for param in self.parameters():
            param.requires_grad = False

    def __str__(self):
        """Return string description of the model"""
        return self.description


def load_random_state(random_state):
    """Method to load the random state"""
    np.random.set_state(random_state["np"])
    random.setstate(random_state["python"])
    torch.set_rng_state(random_state["pytorch"])