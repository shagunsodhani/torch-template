"""Function to get the config id"""
import argparse


def argument_parser() -> str:
    """Function to get the config id"""
    parser = argparse.ArgumentParser(
        description="Argument parser to obtain the name of the config file"
    )
    parser.add_argument(
        "--config_id",
        default="sample_config",
        help="config id to use",
    )
    args = parser.parse_args()
    assert isinstance(args.config_id, str)
    return args.config_id
