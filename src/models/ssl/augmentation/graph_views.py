""" Many graph view functions can be drawing straight from dig
dig.sslgraph.method.contrastive.views_fn
"""

from models.ssl.base.augmentation import ViewAugmentor
from dig.sslgraph.method.contrastive.views_fn import EdgePerturbation, NodeAttrMask, UniformSample, RWSample

class DigEdgePerturbation(ViewAugmentor):
    """ Wrapper for DIG's EdgePerturbation in line with my ABC
    """
    def __init__(self, add=True, drop=False, ratio=0.1):
        self.add = add
        self.drop = drop
        self.ratio = ratio

    def aug_func(self, data):
        dig = EdgePerturbation(self.add, self.drop, self.ratio)
        return dig.do_trans(data)
    
class DigNodeAttrMask(ViewAugmentor):
    def __init__(self, mode="whole", mask_mean=0.5, mask_std=0.5, ratio=0.1):
        self.mode = mode
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.ratio = ratio

    def aug_func(self, data):
        dig = NodeAttrMask(mode=self.mode, mask_ratio=self.ratio, mask_mean=self.mask_mean, mask_std=self.mask_std, return_mask=False)
        return dig.do_trans(data)
    
class DigNodeDropping(ViewAugmentor):
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def aug_func(self, data):
        dig = UniformSample(ratio=self.ratio)
        return dig.do_trans(data)
    
class DigSubgraph(ViewAugmentor):
    def __init__(self, ratio=0.1, add_self_loop=False):
        self.ratio = ratio
        self.add_self_loop = add_self_loop

    def aug_func(self, data):
        dig = RWSample(ratio=self.ratio, add_self_loop=self.add_self_loop)
        return dig.do_trans(data)
