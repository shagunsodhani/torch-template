from abc import ABC, abstractmethod
from typing import Any, Optional


class Checkpointable(ABC):
    """This abstract class provides two methods: (i) save, (ii) load"""

    @abstractmethod
    def save(self, epoch: int) -> Any:
        """Persist the given object to the file system"""
        pass

    @abstractmethod
    def load(self, epoch: Optional[int]) -> Any:
        """Load the given object from the filesystem"""
        pass
