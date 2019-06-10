"""This is the script to uplod the logs to wandb"""

import argparse

from codes.logbook.logbook import LogBook
from codes.utils.config import get_config_from_log
from codes.utils.log import read_log
from codes.utils.util import timing


def is_new_run(log):
    """Method to check if the current log-line indicates a new run"""
    if log and log["type"] == "status" and log["message"].startswith("Starting"):
        return True
    return False


def is_config(log):
    """Method to check if the current log-line is a config log"""
    if log and log["type"] == "config":
        return True
    return False


def is_metric(log):
    """Method to check if the current log-line is a metric log"""
    if log and log["type"] == "metric":
        return True
    return False


@timing
def upload_logs(log_file_path):
    """Run the code"""
    logbook = None
    flag_for_new_run = False
    with open(log_file_path) as log_files:
        for line in log_files:
            log = read_log(line)
            if not log:
                continue
            if is_new_run(log):
                flag_for_new_run = True
                del logbook
            elif is_config(log) and flag_for_new_run:
                log.pop("type")

                log["remote_logger"]["should_use"] = True
                # config = get_frozen_config_box(log)
                config = get_config_from_log(log)
                logbook = LogBook(config)

                flag_for_new_run = False
            elif is_metric(log):
                logbook.write_metric_logs(**log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description') # pylint: disable=invalid-name
    parser.add_argument('config_id', type=str,
                        help='A required integer positional argument') # pylint: disable=invalid-name

    args = parser.parse_args() # pylint: disable=invalid-name

    config_id = args.config_id # pylint: disable=invalid-name

    file_paths = ["/home/t-shsodh/projects/bellman/shagun/logs/{}/log.txt".format(config_id)] # pylint: disable=invalid-name

    for file_path in file_paths:
        upload_logs(log_file_path=file_path)
        # time.sleep(30)
