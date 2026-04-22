from abc import ABC, abstractmethod
from torch_geometric.data import Data, Batch

""" Augmentation functions - converting a single graph into multiple views
"""
class ViewAugmentor(ABC):
    """ classes should be stateless and lightweight
    """
    def __call__(self, data):
        if isinstance(data, Batch):
            dlist = [self.views_func(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        
        elif isinstance(data, Data):
            return self.views_func(data)
        
        else:
            raise TypeError(
                f"Unsupported input type for augmentation: {type(data)}"
            )

    @abstractmethod
    def views_func(self, data: Data) -> Data:
        """ Returns augmented view of a graph
        """
        pass
