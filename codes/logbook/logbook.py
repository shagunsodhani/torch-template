"""Wrapper over wandb api"""

import wandb
from tensorboardX import SummaryWriter

from codes.utils import log as log_func
from codes.utils.util import flatten_dict


class LogBook():
    """Wrapper over comet_ml api"""

    def __init__(self, config):
        self._experiment_id = config.experiment_id
        self.metrics_to_record = \
            [
                "mode ",
                "num_timesteps"
            ]

        flattened_config = flatten_dict(config, sep="_")

        self.should_use_remote_logger = config.logger.remote.should_use

        if self.should_use_remote_logger:
            wandb.init(config=flattened_config,
                       project=config.logger.project_name,
                       name=config.general.id,
                       dir=config.log.dir)

        self.tensorboard_writer = None
        self.should_use_tb = config.logger.tensorboard.should_use
        if self.should_use_tb:
            self.tensorboard_writer = SummaryWriter(comment=config.logger.project_name)

    def log_metrics(self, dic, prefix, step):
        """Method to log metric"""
        formatted_dict = {}
        for key, val in dic.items():
            formatted_dict[prefix + "_" + key] = val
        if self.should_use_remote_logger:
            wandb.log(formatted_dict, step=step)

    def write_config_log(self, config):
        """Write config"""
        log_func.write_config_log(config)
        flatten_config = flatten_dict(config, sep="_")
        flatten_config['experiment_id'] = self._experiment_id

    def write_metric_logs(self, metrics):
        """Write Metric"""
        metrics['experiment_id'] = self._experiment_id
        log_func.write_metric_logs(metrics)
        flattened_metrics = flatten_dict(metrics, sep="_")

        metric_dict = {
            key: flattened_metrics[key]
            for key in self.metrics_to_record if key in flattened_metrics
        }
        prefix = metrics.get("mode", None)
        num_timesteps = metric_dict.pop("num_timesteps")
        self.log_metrics(dic=metric_dict,
                         prefix=prefix,
                         step=num_timesteps)

        if self.should_use_tb:

            timestep_key = "num_timesteps"
            for key in set(list(metrics.keys())) - set([timestep_key]):
                self.tensorboard_writer.add_scalar(tag=key,
                                                   scalar_value=metrics[key],
                                                   global_step=metrics[timestep_key])

    def write_compute_logs(self, **kwargs):
        """Write Compute Logs"""
        kwargs['experiment_id'] = self._experiment_id
        log_func.write_metric_logs(**kwargs)
        metric_dict = flatten_dict(kwargs, sep="_")

        num_timesteps = metric_dict.pop("num_timesteps")
        self.log_metrics(dic=metric_dict,
                         step=num_timesteps,
                         prefix="compute")

    def write_message_logs(self, message):
        """Write message"""
        log_func.write_message_logs(message, experiment_id=self._experiment_id)

    def write_metadata_logs(self, **kwargs):
        """Write metadata"""
        log_func.write_metadata_logs(**kwargs)
        # self.log_other(key="best_epoch_index", value=kwargs["best_epoch_index"])

    # def write_assets(self, **kwargs):
    #     """Write assets"""
    #     self.log_asset(file_path=kwargs["file_path"],
    #                    file_name=kwargs["file_name"],
    #                    overwrite=False)

    # def write_model_graph(self, graph):
    #     """Write model graph"""
    #     self.set_model_graph(self, graph)
