# -*- coding: utf-8 -*-

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
from torch import nn

from mmvae_hub.utils.Dataclasses.Dataclasses import PlanarFlowParams


class AffineFlow(nn.Module):
    """Affine coupling Flow"""

    def __init__(self, class_dim, num_flows, coupling_dim, nbr_coupling_block_layers:int):
        super().__init__()
        self.nbr_coupling_block_layers = nbr_coupling_block_layers
        self.num_flows = num_flows
        self.coupling_dim = coupling_dim
        if num_flows > 0:
            # a simple chain of operations is collected by ReversibleSequential
            # see here for more details: https://vll-hd.github.io/FrEIA/_build/html/FrEIA.modules.html#coupling-blocks
            self.flow = Ff.SequenceINN(class_dim)
            for _ in range(num_flows):
                self.flow.append(Fm.AllInOneBlock, subnet_constructor=self.subnet_fc, permute_soft=True)

    def forward(self, z0, flow_params=None):
        if self.num_flows == 0:
            return z0, torch.zeros_like(z0)
        zk, log_det_jacobian = self.flow(z0)
        return zk, log_det_jacobian

    def rev(self, zk):
        return self.flow(zk, rev=True)

    def get_flow_params(self, h=None):
        # for compat with amortized flows
        return PlanarFlowParams(**{k: None for k in ['u', 'w', 'b']})

    def subnet_fc(self, dims_in, dims_out):
        block = [nn.Linear(dims_in, self.coupling_dim), nn.ReLU()]
        for _ in range(self.nbr_coupling_block_layers):
            block.extend([nn.Linear(self.coupling_dim, self.coupling_dim), nn.ReLU()])
        block.append(nn.Linear(self.coupling_dim, dims_out))
        return nn.Sequential(*block)
