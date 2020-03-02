from typing import Optional

from codes.model.types import OptimizerType, SchedulerType


class OptimizerSchedulerTuple:
    """Class to hold a tuple of optimizer and scheduler
    """

    def __init__(self, optimizer: OptimizerType, scheduler: Optional[SchedulerType]):
        self.optimizer = optimizer
        self.scheduler = scheduler
