"""Class to run the experiments"""
import os
from time import time

import torch

import codes.experiment.metric as metric
from codes.logbook.logbook import LogBook
from codes.utils.util import get_cpu_stats


class Experiment():
    """Experiment Class"""

    def __init__(self, config, agents):
        self.config = config
        self.agents = agents
        self.logbook = LogBook(self.config)
        self.support_modes = self.config.model.modes
        self.device = self.config.general.device
        self.reset_experiment()
        self.startup_logs()
        # torch.autograd.set_detect_anomaly(mode=True)

    def reset_experiment(self):
        """Reset the experiment"""
        pass

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

    def write_git_metadata(self, **kwargs):
        """Method to interface with the logbook"""
        return self.logbook.set_git_metadata()

    def write_message_logs(self, message):
        """Method to interface with the logbook"""
        return self.logbook.write_message_logs(message)

    def write_trajectory_logs(self, trajectory):
        """Method to interface with the logbook"""
        return self.logbook.write_trajectory_logs(trajectory)

    def write_metadata_logs(self, **kwargs):
        """Method to interface with the logbook"""
        return self.logbook.write_metadata_logs(**kwargs)

    def write_assets(self, kwargs):
        """Method to interface with the logbook"""
        return self.logbook.write_assets(**kwargs)

    def write_model_graph(self, graph):
        """Write model graph"""
        self.logbook.write_model_graph(self, graph)

    def set_eval_mode(self):
        """Prepare for the eval mode"""
        pass

    def set_train_mode(self):
        """Prepare for the train mode"""
        pass

    def run(self):
        """Method to run the experiment"""

        start_time = time()

        total_num_steps = 0

        compute_stats = {}
        mode = "train"
        should_log_compute_stats = self.config.log.should_log_compute_stats

        current_metric_dict = metric.get_default_metric_dict(mode=mode)

        for epochs in range(num_updates):

            if should_log_compute_stats:
                compute_stats["start_stats"] = get_cpu_stats()

            self.save(epochs=epochs)

            end_time = time()

            if epochs % self.config.cometml.frequency == 0:
                self.write_metric_logs(**(
                    metric.prepare_metric_dict_to_log(current_metric_dict,
                                                      num_updates=self.config.cometml.frequency)))  # pylint: disable=E1121
                current_metric_dict = metric.get_default_metric_dict(mode)

            if should_log_compute_stats:
                compute_stats["post_training_stats"] = get_cpu_stats()

            evaluate_frequency = self.config.model.evaluate_frequency
            if (evaluate_frequency > 0
                    and epochs % evaluate_frequency == 0):
                with torch.no_grad():
                    self.set_eval_mode()
                    self.evaluate(num_training_steps_so_far=total_num_steps,
                                  num_training_epochs_so_far=epochs)
                    self.set_train_mode()

            if should_log_compute_stats:
                compute_stats["post_eval_stats"] = get_cpu_stats()
                compute_stats["num_timesteps"] = total_num_steps
                self.write_compute_logs(**compute_stats)

            start_time = time()

    def evaluate(self, num_training_steps_so_far, num_training_epochs_so_far):
        """Method to run the experiment"""

        metric_dict = dict(
            mode="test",

        )
        self.write_metric_logs(**metric_dict)  # pylint: disable=E1121

    def startup_logs(self):
        """Method to write some startup logs"""
        self.write_config_log(self.config)
        # self.write_git_metadata()
        self.log_env_file()
        self.log_config_file()

    def log_config_file(self):
        split_key = "codes/experiment"
        current_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(current_path.split(split_key)[0],
                                 "config/{}.yaml".format(self.config.general.id)
                                 )
        self.write_assets(dict(
            file_path=file_path,
            file_name=None,
        ))

    def save(self, epochs):
        """Method to save the experiment"""
        if self.config.model.persist_frquency > 0 \
                and epochs % self.config.model.persist_frquency == 0:
            for agent in self.agents.iterate_over_agents():
                agent.save(epochs)


def prepare_and_run_experiment(config):
    """Primary method to interact with the Experiments"""
    experiment = Experiment(config)
    experiment.run()
