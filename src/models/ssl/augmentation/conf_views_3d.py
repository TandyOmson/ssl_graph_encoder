""" Alternative views are extra higher energy conformers of a molecule
"""

from models.ssl.base.augmentation import ViewAugmentor
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
            return copy.deepcopy(data)
        return random.choice(conformers)