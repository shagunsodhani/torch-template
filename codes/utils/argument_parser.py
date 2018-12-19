"""Function to get the config id"""
import argparse


def argument_parser():
    """Function to get the config id"""
    parser = argparse.ArgumentParser(
        description="Argument parser to obtain the name of the config file")
    parser.add_argument('--config_id', default="sample", help='config id to use')
    args = parser.parse_args()
    return args.config_id
