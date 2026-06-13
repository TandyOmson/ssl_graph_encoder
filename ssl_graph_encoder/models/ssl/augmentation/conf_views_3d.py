""" Alternative views are extra higher energy conformers of a molecule
"""

from ssl_graph_encoder.models.ssl.base.augmentation import ViewAugmentor
import torch
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
            # case of only a single conformer, sometimed perturb the data, sometimes return a positive view
            if random.random() < 0.5:
                eps = random.uniform(0.03, 0.07)
                new_data = self.perturb_positions(data, eps)
            else:
                new_data = data.clone()
            if 'y' in new_data:
                del new_data.y
            return new_data
        
        idx = random.choice(conformers)
        return self.conf_dataset[idx]
    
    @staticmethod
    def perturb_positions(data, eps=0.05):
        """ Perturb positions slightly in the case of only one conformer
            Adds meaningful signal for contrastive loss
            Later on I could make eps adapt to the complexity of the molecule?
        """
        new_data = data.clone()
        noise = torch.randn_like(new_data.pos) * eps
        new_data.pos = new_data.pos + noise
        return new_data
