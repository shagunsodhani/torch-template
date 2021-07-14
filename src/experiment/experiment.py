"""Class to interface with an Experiment"""

from __future__ import print_function

import os
from time import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.data
from xplogger import logbook
from xplogger import metrics as ml_metrics
from xplogger.types import LogType

from src.dataset import builder as dataset_builder
from src.dataset.types import TensorType
from src.model import builder as model_builder
from src.model.base_model import BaseModel
from src.model.utils import OptimizerSchedulerTuple
from src.utils import config as config_utils
from src.utils.checkpointable import Checkpointable
from src.utils.config import ConfigType


class Experiment(Checkpointable):
    """Experiment Class"""

    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        self.id = experiment_id
        self.config = config
        logbook_config = logbook.make_config(
            logger_file_path=self.config.logbook.logger_file_path,
            tensorboard_config=self.config.logbook.tensorboard,
        )
        self.logbook = logbook.LogBook(config=logbook_config)
        self.device = torch.device(self.config.general.device)
        self.dataloaders = dataset_builder.build(config=self.config)
        self.model = model_builder.build(config=self.config, device=self.device)
        assert isinstance(self.model, BaseModel)
        optimizer_and_scheduler = self.model.get_optimizer_and_scheduler()
        self.optimizer = optimizer_and_scheduler.optimizer
        self.scheduler = optimizer_and_scheduler.scheduler

        self.loss_fn = nn.CrossEntropyLoss()

        self.global_metrics = ml_metrics.MetricDict(
            [
                ml_metrics.MinMetric("loss"),
                ml_metrics.MaxMetric("accuracy"),
                ml_metrics.ConstantMetric("mode", "best_performance_on_test"),
                ml_metrics.CurrentMetric("epoch"),
            ]
        )

        self.startup_logs()

    def startup_logs(self) -> None:
        """Method to write some startup logs"""
        self.logbook.write_message("Starting experiment id: {}".format(self.id))
        self.logbook.write_config_log(config_utils.to_dict(self.config))

    def save(self, epoch: int) -> None:
        """Method to save the experiment"""

        self.model.save(  # type: ignore
            epoch=epoch,
            optimizer_scheduler_tuple=OptimizerSchedulerTuple(
                optimizer=self.optimizer, scheduler=self.scheduler
            ),
            is_best_model=False,
            path_to_save_to=os.path.join(self.config.experiment.save_dir, self.id),
        )

    def _load_components(self, path_to_load_from: str, epoch: int) -> None:
        optimizer_scheduler_tuple, message = self.model.load(  # type: ignore
            epoch=epoch,
            optimizer_scheduler_tuple=OptimizerSchedulerTuple(
                optimizer=self.optimizer, scheduler=self.scheduler
            ),
            path_to_load_from=path_to_load_from,
        )
        self.optimizer = optimizer_scheduler_tuple.optimizer
        self.scheduler = optimizer_scheduler_tuple.scheduler
        self.logbook.write_message(message)

    def load(self, epoch: Optional[int]) -> int:
        """Method to load the entire experiment"""

        if epoch is None:
            path_to_load_epoch_state = os.path.join(
                self.config.experiment.save_dir, self.id, "epoch.tar"
            )
            self.logbook.write_message(
                f"Considering {path_to_load_epoch_state} to load epoch state for {self.id}"
            )
            if not os.path.exists(path_to_load_epoch_state):
                # New Experiment
                self.logbook.write_message(
                    f"""This is a new experiment. Can not load any model
                    from {path_to_load_epoch_state} for experiment {self.id}"""
                )
                return -1
            epoch_state = torch.load(path_to_load_epoch_state)  # type: ignore
            epoch = epoch_state["current"]
        path_to_load_from = os.path.join(
            self.config.experiment.save_dir, self.id, f"epoch_{epoch}.tar"
        )
        self._load_components(path_to_load_from=path_to_load_from, epoch=epoch)
        self.logbook.write_message(
            f"Loading the model for experiment {self.id} from {path_to_load_from}"
        )
        return epoch + 1

    def periodic_save(self, epoch: int) -> None:
        """Method to perioridically save the experiment.
        This method is a utility method, built on top of save method.
        It performs an extra check of wether the experiment is configured to
        be saved during the current epoch."""
        persist_frequency = self.config.experiment.persist_frequency
        if persist_frequency > 0 and epoch % persist_frequency == 0:
            self.save(epoch)

    def run(self) -> None:
        start_epoch = 0
        for epoch in range(
            start_epoch, start_epoch + self.config.experiment.num_epochs
        ):
            self.train(epoch)
            self.test(epoch)
            if self.scheduler:
                self.scheduler.step()  # type: ignore
        self.periodic_save(epoch)

    def train(self, epoch: int) -> None:
        epoch_start_time = time()
        self.model.train()
        mode = "train"
        metric_dict = init_metric_dict(epoch=epoch, mode=mode)
        trainloader = self.dataloaders[mode]
        for batch_idx, batch in enumerate(trainloader):  # noqa: B007
            current_metric = self.compute_metrics_for_batch(batch=batch, mode=mode)
            metric_dict.update(metrics_dict=current_metric)
        metric_dict = metric_dict.to_dict()
        metric_dict["time_taken"] = time() - epoch_start_time
        self.logbook.write_metric_log(metric=prepare_metric_dict_for_tb(metric_dict))

    def test(self, epoch: int) -> None:
        epoch_start_time = time()
        self.model.eval()
        mode = "test"
        metric_dict = init_metric_dict(epoch=epoch, mode=mode)
        testloader = self.dataloaders[mode]
        for batch_idx, batch in enumerate(testloader):  # noqa: B007
            with torch.no_grad():
                current_metric = self.compute_metrics_for_batch(batch=batch, mode=mode)
            metric_dict.update(metrics_dict=current_metric)
        metric_dict = metric_dict.to_dict()
        metric_dict["time_taken"] = time() - epoch_start_time
        self.global_metrics.update(metrics_dict=metric_dict)
        self.logbook.write_metric_log(metric=prepare_metric_dict_for_tb(metric_dict))
        self.logbook.write_metric_log(
            metric=prepare_metric_dict_for_tb(self.global_metrics.to_dict())
        )

    def compute_metrics_for_batch(
        self, batch: Tuple[TensorType, TensorType], mode: str
    ) -> LogType:
        should_train = mode == "train"
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        if should_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        _, predicted = outputs.max(1)
        num_correct = predicted.eq(targets).sum().item()
        total = targets.size(0)

        current_metric = {
            "loss": loss.item(),
            "accuracy": num_correct * 1.0 / total,
        }
        return current_metric


def init_metric_dict(epoch: int, mode: str) -> ml_metrics.MetricDict:
    metric_dict = ml_metrics.MetricDict(
        [
            ml_metrics.AverageMetric("loss"),
            ml_metrics.AverageMetric("accuracy"),
            ml_metrics.ConstantMetric("epoch", epoch),
            ml_metrics.ConstantMetric("mode", mode),
        ]
    )
    return metric_dict


def prepare_metric_dict_for_tb(metric: LogType) -> LogType:
    metric["main_tag"] = metric["mode"]
    metric["global_step"] = metric["epoch"]
    return metric
