# -*- coding: utf-8 -*-

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
from torch import nn

from mmvae_hub.utils.Dataclasses.Dataclasses import PlanarFlowParams


class ConvFlow(nn.Module):
    """Convolutional coupling Flow"""

    def __init__(self, class_dim, num_flows, coupling_dim):
        super().__init__()
        self.num_flows = num_flows
        self.coupling_dim = coupling_dim
        self.flow = self.get_flow(embedding_dim=class_dim, num_residual_hiddens=32)

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

    def get_flow(self, embedding_dim, num_residual_hiddens):
        nodes = [Ff.InputNode(embedding_dim, num_residual_hiddens, num_residual_hiddens, name='input')]

        def subnet_conv(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, self.coupling_dim, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv2d(self.coupling_dim, c_out, kernel_size=3, padding=1))

        # Higher resolution convolutional part
        for k in range(self.num_flows):
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet_conv, 'clamp': 1.2},
                                 name=F'conv_high_res_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                 Fm.PermuteRandom,
                                 {'seed': k},
                                 name=F'permute_high_res_{k}'))

        # nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))

        return Ff.GraphINN(nodes)


if __name__ == '__main__':
    flow = ConvFlow(class_dim=64, num_flows=5, coupling_dim=256)
    inp = torch.rand((2, 64, 32, 32))
    print(inp.mean())
    a = flow.flow(inp)
    print(a[0].mean())
    print(a[0].shape)
    b,c  = flow.rev(inp)
    dsfsd = 0
