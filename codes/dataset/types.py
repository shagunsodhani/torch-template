from typing import TYPE_CHECKING

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

TensorType = Tensor

if TYPE_CHECKING:
    DataLoaderType = DataLoader[TensorType]
    DatasetType = Dataset[TensorType]
else:
    DataLoaderType = DataLoader
    DatasetType = Dataset
