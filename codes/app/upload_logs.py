"""This is the script to uplod the logs to wandb"""

from codes.logbook.logbook import LogBook
from codes.utils.log import read_log
from codes.utils.serializable_config import get_frozen_config_box
from codes.utils.config import get_config_from_log
from codes.utils.util import timing
import time
import argparse

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
def upload_logs(file_path):
    """Run the code"""
    logbook = None
    flag_for_new_run = False
    with open(file_path) as f:
        for line in f:
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
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('config_id', type=str,
                       help='A required integer positional argument')

    args = parser.parse_args()

    config_id = args.config_id

    file_paths = [
    "/home/t-shsodh/projects/bellman/shagun/logs/{}/log.txt".format(config_id)
    ]
    for file_path in file_paths:
        upload_logs(file_path)
        # time.sleep(30)
