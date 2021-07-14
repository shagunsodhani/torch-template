"""This is the main entry point for the code"""

from src.app import utils
from src.experiment import utils as experiment_utils
from src.utils.argument_parser import argument_parser


def run(config_id: str) -> None:
    """Run the code"""

    config = utils.bootstrap_config(config_id)
    experiment_utils.prepare_and_run(config=config)


if __name__ == "__main__":
    run(config_id=argument_parser())
