import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

nodes = [Ff.InputNode(3, 32, 32, name='input')]
ndim_x = 3 * 32 * 32


def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(256, c_out, 3, padding=1))


# Higher resolution convolutional part
for k in range(4):
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

conv_inn = Ff.GraphINN(nodes)

conv_inn(torch.rand((1, 3, 32, 32)))
