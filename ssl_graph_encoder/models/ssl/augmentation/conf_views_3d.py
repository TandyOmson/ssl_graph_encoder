""" Alternative views are extra higher energy conformers of a molecule
"""

from ssl_graph_encoder.models.ssl.base.augmentation import ViewAugmentor
import random
import copy

class AlternativeConformer(ViewAugmentor):
    """ Picks a random conformer from alt_conf_data (already data objects)
    """
    def __init__(self, alt_conf_dict):
        self.alt_conf_dict = alt_conf_dict
        
    def aug_func(self, data):        
        conformers = self.alt_conf_dict[int(data.sample_id)]
        # edge where there is only one stable conformer
        if not conformers:
            new_data = copy.deepcopy(data)
            if 'y' in new_data:
                del new_data.y
            return new_data
        return random.choice(conformers)