""" Alternative views are extra higher energy conformers of a molecule
"""

from ssl_graph_encoder.models.ssl.base.augmentation import ViewAugmentor
import random

class AlternativeConformer(ViewAugmentor):
    """ Picks a random conformer from alt_conf_data (already data objects)
    """
    def __init__(self, alt_conf_dict, conf_dataset):
        self.alt_conf_dict = {
            int(k): v for k, v in alt_conf_dict.items()
        }
        self.conf_dataset = conf_dataset
        
    def aug_func(self, data):        
        try:
            conformers = self.alt_conf_dict[int(data.sample_id)]
        except KeyError:
            new_data = data.clone()
            if 'y' in new_data:
                del new_data.y
            return new_data
        
        idx = random.choice(conformers)
        return self.conf_dataset[idx]
