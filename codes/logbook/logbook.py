"""Wrapper over wandb api"""

import json

import wandb
from tensorboardX import SummaryWriter

from codes.logbook import filesystem_logger as fs_log
from codes.utils.util import flatten_dict, make_dir


class LogBook:
    """Wrapper over comet_ml api"""

    def __init__(self, config):
        self._experiment_id = config.general.experiment_id
        self.metrics_to_record = ["mode ", "num_timesteps"]

        flattened_config = flatten_dict(config.to_serializable_dict(), sep="_")

        self.should_use_remote_logger = config.logger.remote.should_use

        if self.should_use_remote_logger:
            wandb.init(
                config=flattened_config,
                notes=config.general.description,
                project=config.logger.remote.project_name,
                name=config.general.id,
                entity=config.logger.remote.entity,
                dir=config.logger.file.dir,
            )

            dir_to_save_config = f"{wandb.run.dir}/config"
            make_dir(dir_to_save_config)

            with open(f"{dir_to_save_config}/{config.general.id}.yaml", "w") as f:
                f.write(json.dumps(config.to_serializable_dict(), indent=4))

        self.tensorboard_writer = None
        self.should_use_tb = config.logger.tensorboard.should_use
        if self.should_use_tb:
            self.tensorboard_writer = SummaryWriter(comment=config.logger.project_name)

    def _log_metrics(self, dic, prefix, step):
        """Method to log metric"""
        formatted_dict = {}
        for key, val in dic.items():
            formatted_dict[prefix + "_" + key] = val
        if self.should_use_remote_logger:
            wandb.log(formatted_dict, step=step)

    def write_config_log(self, config):
        """Write config"""
        fs_log.write_config_log(config)
        flatten_config = flatten_dict(config, sep="_")
        flatten_config["experiment_id"] = self._experiment_id

    def write_metric_logs(self, metrics):
        """Write Metric"""
        metrics["experiment_id"] = self._experiment_id
        fs_log.write_metric_logs(metrics)
        flattened_metrics = flatten_dict(metrics, sep="_")

        metric_dict = {
            key: flattened_metrics[key]
            for key in self.metrics_to_record
            if key in flattened_metrics
        }
        prefix = metrics.get("mode", None)
        num_timesteps = metric_dict.pop("num_timesteps")
        self._log_metrics(dic=metric_dict, prefix=prefix, step=num_timesteps)

        if self.should_use_tb:

            timestep_key = "num_timesteps"
            for key in set(list(metrics.keys())) - set([timestep_key]):
                self.tensorboard_writer.add_scalar(
                    tag=key,
                    scalar_value=metrics[key],
                    global_step=metrics[timestep_key],
                )

    def write_compute_logs(self, **kwargs):
        """Write Compute Logs"""
        kwargs["experiment_id"] = self._experiment_id
        fs_log.write_metric_logs(**kwargs)
        metric_dict = flatten_dict(kwargs, sep="_")

        num_timesteps = metric_dict.pop("num_timesteps")
        self._log_metrics(dic=metric_dict, step=num_timesteps, prefix="compute")

    def write_message_logs(self, message):
        """Write message logs"""
        fs_log.write_message_logs(message, experiment_id=self._experiment_id)

    def write_metadata_logs(self, metadata):
        """Write metadata"""
        metadata["experiment_id"] = self._experiment_id
        fs_log.write_metadata_logs(metadata)
        # self.log_other(key="best_epoch_index", value=kwargs["best_epoch_index"])

    def watch_model(self, model):
        """Method to track the gradients of the model"""
        wandb.watch(models=model, log="all")
