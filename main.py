"""This is the main entry point for the code."""

import hydra
from codes.experiment import utils as experiment_utils
from codes.utils.types import ConfigType


@hydra.main(config_path="config", config_name="config")  # type: ignore
def run(config: ConfigType) -> None:
    print(config.pretty())
    experiment_utils.prepare_and_run(config=config)


if __name__ == "__main__":
    run()
