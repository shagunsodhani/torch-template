"""Wrapper over comet_ml api"""
from comet_ml import Experiment

from codes.utils import log as log_func
from codes.utils.util import flatten_dict


class LogBook(Experiment):
    """Wrapper over comet_ml api"""

    def __init__(self, config):
        self._experiment_id = config.general.experiment_id
        self.metrics_to_record = [
            "mode",
            "num_timesteps",
        ]

        super().__init__(
            api_key=config.cometml.api_key,
            project_name=config.cometml.project_name,
            workspace=config.cometml.workspace,
            disabled=not config.cometml.should_use,
            log_code=False,
            parse_args=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_output_logging=None,
            log_env_details=True,
            log_git_metadata=True,
        )

        # self.set_filename("{}".format(config.general.id))
        self.set_name("{}".format(config.general.id))

    def write_config_log(self, config):
        """Write config"""
        log_func.write_config_log(config)
        flatten_config = flatten_dict(config, sep="_")
        flatten_config['experiment_id'] = self._experiment_id
        self.log_parameters(dic=flatten_config)

    def write_experiment_name(self, config):
        """Write config"""
        self.set_name(name=config.general.id)

    def write_metric_logs(self, **kwargs):
        """Write Metric"""
        kwargs['experiment_id'] = self._experiment_id
        log_func.write_metric_logs(**kwargs)
        flattened_metrics = flatten_dict(kwargs, sep="_")

        metric_dict = {
            key: flattened_metrics[key]
            for key in self.metrics_to_record if key in flattened_metrics
        }
        prefix = kwargs['mode']
        num_timesteps = metric_dict.pop("num_timesteps")
        self.log_metrics(dic=metric_dict,
                         prefix=prefix,
                         step=num_timesteps)

    def write_compute_logs(self, **kwargs):
        """Write Compute Logs"""
        kwargs['experiment_id'] = self._experiment_id
        log_func.write_metric_logs(**kwargs)
        metric_dict = flatten_dict(kwargs, sep="_")

        num_timesteps = metric_dict.pop("num_timesteps")
        self.log_metrics(dic=metric_dict,
                         step=num_timesteps)

    def write_message_logs(self, message):
        """Write message"""
        log_func.write_message_logs(message, experiment_id=self._experiment_id)
        # self.log_other(key="message", value="{}_{}".format(message, self._experiment_id))

    def write_trajectory_logs(self, trajectory):
        """Write message"""
        log_func.write_trajectory_logs(trajectory, experiment_id=self._experiment_id)

    def write_metadata_logs(self, **kwargs):
        """Write metadata"""
        log_func.write_metadata_logs(**kwargs)
        self.log_other(key="best_epoch_index", value=kwargs["best_epoch_index"])

    def write_assets(self, **kwargs):
        """Write assets"""
        self.log_asset(file_path=kwargs["file_path"],
                       file_name=kwargs["file_name"],
                       overwrite=False)

    def write_model_graph(self, graph):
        """Write model graph"""
        self.set_model_graph(self, graph)
