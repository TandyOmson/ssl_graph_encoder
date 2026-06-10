from abc import ABC, abstractmethod
from torch_geometric.data import Data

""" Augmentation functions - converting a single graph into multiple views
"""
class ViewAugmentor(ABC):
    """ classes should be stateless and lightweight
    """
    def __call__(self, data):
        if not isinstance(data, Data):
            raise TypeError(
                f"Unsupported input type for augmentation: {type(data)}"
            )
        return self.aug_func(data)

    @abstractmethod
    def aug_func(self, data: Data) -> Data:
        """ Returns augmented view of a graph
        """
        pass
