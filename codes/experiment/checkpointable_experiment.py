"""Class to run the experiments"""
# from time import time

import torch
import os

from codes.utils.checkpointable import Checkpointable
from typing import Optional, Iterable

from codes.experiment import metric
from codes.logbook.logbook import LogBook
from codes.utils.util import get_cpu_stats


class CheckpointableExperiment(Checkpointable):
    """Checkpointable Experiment Class"""

    def __init__(self, config, model):
        self.config = config
        self.logbook = LogBook(self.config)
        self.support_modes = self.config.model.modes
        self.device = self.config.general.device
        self.model = model
        self.logbook.watch_model(model=self.model)
        self.optimizer, self.scheduler = self.get_scheduler_and_optimizer()
        self.epoch = None
        self.logbook.watch_model(model=self.model)
        self._mode = None
        self.dataloaders = get_dataloaders(
            config=config,
            modes=["train", "val", "test"]
        )
        self.reset_experiment()
        self.startup_logs()
        # torch.autograd.set_detect_anomaly(mode=True)

    def reset_experiment(self):
        """Reset the experiment"""
        self._mode = None
        self.epoch = 0

    def get_scheduler_and_optimizer(self):
        optimizers, schedulers = self.model.get_optimizers_and_schedulers()
        return optimizers[0], schedulers[0]

    def write_config_log(self, config):
        """Method to interface with the logbook"""
        return self.logbook.write_config_log(config)

    def write_experiment_name(self, config):
        """Method to interface with the logbook"""
        return self.logbook.write_config_log(config)

    def write_metric_logs(self, **kwargs):
        """Method to interface with the logbook"""
        return self.logbook.write_metric_logs(**kwargs)

    def write_compute_logs(self, **kwargs):
        """Method to interface with the logbook"""
        return self.logbook.write_compute_logs(**kwargs)

    # def write_git_metadata(self):
    #     """Method to interface with the logbook"""
    #     return self.logbook.set_git_metadata()

    def write_message_logs(self, message):
        """Method to interface with the logbook"""
        return self.logbook.write_message_logs(message)

    # def write_trajectory_logs(self, trajectory):
    #     """Method to interface with the logbook"""
    #     return self.logbook.write_trajectory_logs(trajectory)

    def write_metadata_logs(self, **kwargs):
        """Method to interface with the logbook"""
        return self.logbook.write_metadata_logs(**kwargs)

    # def write_assets(self, kwargs):
    #     """Method to interface with the logbook"""
    #     return self.logbook.write_assets(**kwargs)

    # def write_model_graph(self, graph):
    #     """Write model graph"""
    #     self.logbook.write_model_graph(graph)

    def set_eval_mode(self):
        """Prepare for the eval mode"""
        pass  # pylint: disable=W0107

    def set_train_mode(self):
        """Prepare for the train mode"""
        pass  # pylint: disable=W0107

    def run(self):
        """Method to run the experiment"""

        # start_time = time()

        compute_stats = {}
        mode = "train"
        should_log_compute_stats = self.config.log.should_log_compute_stats

        current_metric_dict = metric.get_default_metric_dict(mode=mode)

        current_epoch = 0
        epoch_to_start_from = self.epoch

        for current_epoch in range(
            epoch_to_start_from, self.config.model.num_epochs
        ):
           if should_log_compute_stats:
                compute_stats["start_stats"] = get_cpu_stats()
            self.epoch = current_epoch
            self.periodic_save(epoch=current_epoch)

            if self.epoch % self.config.cometml.frequency == 0:
                self.write_metric_logs(**(
                    metric.prepare_metric_dict_to_log(current_metric_dict,
                                                      num_updates=self.config.cometml.frequency)))  # pylint: disable=E1121
                current_metric_dict = metric.get_default_metric_dict(mode)

            if should_log_compute_stats:
                compute_stats["post_training_stats"] = get_cpu_stats()

            evaluate_frequency = self.config.model.evaluate_frequency
            if (evaluate_frequency > 0
                    and self.epoch % evaluate_frequency == 0):
                with torch.no_grad():
                    self.set_eval_mode()
                    self.evaluate(num_training_steps_so_far=total_num_steps,
                                  num_training_epochs_so_far=self.epoch)
                    self.set_train_mode()

            if should_log_compute_stats:
                compute_stats["post_eval_stats"] = get_cpu_stats()
                compute_stats["num_timesteps"] = total_num_steps
                self.write_compute_logs(**compute_stats)

            # start_time = time()

    def evaluate(self, num_training_steps_so_far, num_training_epochs_so_far):
        """Method to run the experiment"""

        metric_dict = dict(
            mode="test",
            num_training_steps_so_far=num_training_steps_so_far,
            num_training_epochs_so_far=num_training_epochs_so_far
        )
        self.write_metric_logs(**metric_dict)  # pylint: disable=E1121

    def startup_logs(self):
        """Method to write some startup logs"""
        self.write_config_log(self.config)
        # self.write_git_metadata()
        # self.log_config_file()

    # def log_config_file(self):
    #     """Method to log the config file"""
    #     split_key = "codes/experiment"
    #     current_path = os.path.dirname(os.path.realpath(__file__))
    #     file_path = os.path.join(current_path.split(split_key)[0],
    #                              "config/{}.yaml".format(self.config.general.id)
    #                              )
    #     self.write_assets(dict(
    #         file_path=file_path,
    #         file_name=None,
    #     ))

    def save(self, epoch: Optional[int]) -> None:
        """Method to save the experiment"""
        if epoch is None:
            epoch = self.epoch
        self.model.save(epoch, optimizers=[self.optimizer])

    def load(self, epoch: Optional[int]) -> None:
        """Method to load the entire experiment"""
        if epoch is None:
            path_to_load_epoch_state = os.path.join(
                self.config.model.save_dir, "epoch.tar"
            )
            epoch_state = torch.load(path_to_load_epoch_state)
            epoch = epoch_state["current"]
        self.epoch = epoch
        self._load_model()

    def _load_model(self) -> None:
        """Internal method to load the model (only)"""
        optimizers, schedulers = self.model.load(
            epoch=self.epoch, optimizers=[self.optimizer], schedulers=[self.scheduler]
        )
        self.optimizer = optimizers[0]
        self.scheduler = schedulers[0]

    def periodic_save(self, epoch):
        """Method to perioridically save the experiment.
        This method is a utility method, built on top of save method.
        It performs an extra check of wether the experiment is configured to
        be saved during the current epoch."""
        if (
            self.config.model.persist_frquency > 0
            and epochs % self.config.model.persist_frquency == 0
            and epoch % self.config.model.persist_frquency == 0
        ):
            self.model.save_model(epochs)
            self.save(epoch)


def prepare_and_run_experiment(config, model):
    """Primary method to interact with the Experiments"""
    experiment = CheckpointableExperiment(config, model)
    experiment.run()
