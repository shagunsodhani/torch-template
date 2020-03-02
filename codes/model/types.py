from typing import TYPE_CHECKING, Any, Callable

from torch import nn, optim

from codes.dataset.types import TensorType

if TYPE_CHECKING:
    OptimizerType = optim.optimizer.Optimizer
else:
    OptimizerType = optim.Optimizer
SchedulerType = optim.lr_scheduler._LRScheduler

if TYPE_CHECKING:
    ModelType = nn.Module[TensorType]
else:
    ModelType = Any

LossFunctionType = Callable[[TensorType, TensorType], TensorType]
