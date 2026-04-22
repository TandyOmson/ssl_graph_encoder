from models.ssl.base.loss import ContrastiveLoss
import torch

class InfoNCE(ContrastiveLoss):
    """ InfoNCE
    """
    def __init__(self, temperature=0.5, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        # will need to add zs_n, batch for other losses
        # or even simga (2D array of which constrative loss pairs are calculated)

    def forward(self, zs):
        # currently only implemented for len(zs) == 2 (i.e. two augmented views)
        if len(zs) == 2:
            loss = self.NT_Xent(zs[0], zs[1], self.temperature, self.normalize)
            return loss    
        else:
            raise NotImplementedError
        
    @staticmethod
    def NT_Xent(z1, z2, temperature, normalize):
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum("ik,jk->ij", z1, z2)

        if normalize:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix/temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()

        return loss