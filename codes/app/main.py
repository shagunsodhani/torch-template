"""This is the main entry point for the code"""

from codes.app import utils
from codes.experiment import utils as experiment_utils
from codes.utils.argument_parser import argument_parser


def run(config_id: str) -> None:
    """Run the code"""

    config = utils.bootstrap_config(config_id)
    experiment_utils.prepare_and_run(config=config)


if __name__ == "__main__":
    run(config_id=argument_parser())
