"""GradStudent is the orchestrator of the experiment"""

import time

import torch
import torch.multiprocessing as mp

from codes.experiment.experiment import prepare_and_run_experiment
from codes.logbook.filesystem_logger import set_logger, write_message_logs, write_config_log
from codes.model.base_model import BaseModel
from codes.utils.config import get_config
from codes.utils.util import set_seed


class GradStudent():
    """GradStudent Class

    In practice, it is a thin class to support multiple experiments at once."""

    def __init__(self, config_id):
        self.config = bootstrap_config(config_id)
        self.num_experiments = self.config.general.num_experiments
        torch.set_num_threads(self.num_experiments)  # pylint: disable=E1101
        self.device = self.config.general.device
        self.model = self.bootstrap_model()

    def bootstrap_model(self):
        """Method to instantiate the models that will be common to all the experiments."""
        return BaseModel(self.config)

    def run(self):
        """Method to run the task"""
        if self.num_experiments > 1 and self.device.type != "cpu":
            write_message_logs("Multi GPU training not supported.")
            return

        if self.num_experiments > 1:

            # for model in self.models:
            self.model.share_memory()

            processes = []
            for experiment_id in range(self.num_experiments):
                config = get_config(self.config.general.id, experiment_id=experiment_id)
                proc = mp.Process(target=prepare_and_run_experiment,
                                  args=(config,
                                        self.model))
                proc.start()
                processes.append(proc)
            for proc in processes:
                proc.join()
        else:
            prepare_and_run_experiment(config=self.config,
                                       model=self.model)


def bootstrap_config(config_id):
    """Method to generate the config (using config id) and set seeds"""
    config = get_config(config_id, experiment_id=0)
    print(config)
    set_logger(config)
    write_message_logs("Starting Experiment at {}".
                       format(time.asctime(time.localtime(time.time()))))
    write_message_logs("torch version = {}".format(torch.__version__))
    write_config_log(config)
    set_seed(seed=config.general.seed)
    return config
