"""Base class for all the models (with batteries)"""
import importlib
import os
import random
from time import time

import numpy as np
import torch
from torch import nn, optim

from codes.utils.log import write_message_logs


class BaseModel(nn.Module):
    """Base class for all models"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = "base_model"
        self.description = "This is the base class for all the models. " \
                           "All the other models should extend this class. " \
                           "It is not to be used directly."
        self.criteria = nn.BCEWithLogitsLoss()
        self.epsilon = 1e-6

    def loss(self, outputs, labels):
        """Method to compute the loss"""
        return self.criteria(outputs, labels)

    def track_loss(self, outputs, labels):
        """There are two different functions related to loss as we might be interested in
         tracking one loss and optimising another"""
        return self.loss(outputs, labels)

    def save_model(self, epochs=-1, optimizers=None, schedulers=None,
                   is_best_model=False, index=0):
        """Method to persist the model.
        Note this method is not well tested"""
        model_config = self.config.model
        state = {
            "epochs": epochs + 1,
            "state_dict": self.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "np_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "pytorch_random_state": torch.get_rng_state(),
            "index": index,
            # "schedulers": [scheduler.state_dict() for scheduler in schedulers]
        }
        if is_best_model:
            path = os.path.join(model_config.save_dir,
                                "best",
                                "{}_agent_id_{}.tar".format(
                                    self.config.general.experiment_id,
                                    index))
        else:
            path = os.path.join(model_config.save_dir,
                                str(epochs + 1),
                                "{}_agent_id_{}.tar".format(
                                    self.config.general.experiment_id,
                                    index))

        torch.save(state, path)
        write_message_logs("saved model to path = {}".format(path))

    def load_model(self, index=0, should_load_optimizers=False,
                   optimizers=None, schedulers=None):
        """Method to load the model"""
        model_config = self.config.model
        load_path = model_config.load_path
        if load_path[-1] == "/":
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
        epochs = checkpoint["epochs"]
        load_metadata(checkpoint)
        self._load_model_params(checkpoint["state_dict"])

        if should_load_optimizers:
            if optimizers is None:
                optimizers = self.get_optimizers()
            for optim_index, optimizer in enumerate(optimizers):
                # optimizer.load_state_dict(checkpoint[OPTIMIZERS][optim_index]())
                optimizer.load_state_dict(checkpoint["optimizers"][optim_index])
            # for scheduler_index, scheduler in enumerate(schedulers):
            # optimizer.load_state_dict(checkpoint[OPTIMIZERS][optim_index]())
            # scheduler.load_state_dict(checkpoint["schedulers"][scheduler_index])
        return optimizers, schedulers, epochs

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
        agent = self.config.model.policy.primitive.agent
        optimizer_config = self.config.model[agent].optimizer
        optimizer_cls = getattr(importlib.import_module("torch.optim"), optimizer_config.name)
        if optimizer_config.name == "RMSprop":
            return optimizer_cls(
                model_params,
                alpha=optimizer_config.alpha,
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.l2_penalty,
                eps=optimizer_config.eps
            )

        return optimizer_cls(
            model_params,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.l2_penalty,
            eps=optimizer_config.eps
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

    def forward(self, data): # pylint: disable=W0221
        '''Forward pass of the network'''
        pass

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


def load_metadata(checkpoint):
    """Method to load the model metadata"""
    np.random.set_state(checkpoint["np_random_state"])
    random.setstate(checkpoint["python_random_state"])
    torch.set_rng_state(checkpoint["pytorch_random_state"])
