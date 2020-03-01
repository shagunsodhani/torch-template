from abc import ABC, abstractmethod


class Checkpointable(ABC):
    """This abstract class provides two methods: (i) save=, (ii) load"""

    @abstractmethod
    def save(self, epoch=None, **kwargs):
        """Persist the given object to the file system"""
        pass

    @abstractmethod
    def load(self, epoch=None, **kwargs):
        """Load the given object from the filesystem"""
        pass
