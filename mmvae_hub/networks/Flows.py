# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch
import torch.nn as nn

from mmvae_hub.networks.utils import flows
from mmvae_hub.utils.Dataclasses import *


class Flow(nn.Module):
    def __init__(self, flags):
        super().__init__()
        self.flags = flags

    @abstractmethod
    def forward(self):
        pass


class PlanarFlow(Flow):
    def __init__(self, flags):
        super().__init__(flags)
        self.mm_div = None

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = flags.num_flows

        if self.num_flows:
            # Amortized flow parameters
            if flags.amortized_flow:
                self.amor_u = nn.Linear(flags.class_dim, self.num_flows * flags.class_dim)
                self.amor_w = nn.Linear(flags.class_dim, self.num_flows * flags.class_dim)
                self.amor_b = nn.Linear(flags.class_dim, self.num_flows)
            else:
                self.u = torch.empty((1, self.num_flows, flags.class_dim, 1)).to(self.flags.device).requires_grad_(True)
                torch.nn.init.normal_(self.u, 0, 0.1)

                self.w = torch.empty((1, self.num_flows, 1, flags.class_dim)).to(self.flags.device).requires_grad_(True)
                torch.nn.init.normal_(self.w, 0, 0.1)

                self.b = torch.zeros((1, self.num_flows, 1, 1)).to(self.flags.device).requires_grad_(True)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def forward(self, in_distr: Distr, flow_params: PlanarFlowParams):
        num_samples = in_distr.mu.shape[0]
        log_det_j = torch.zeros(in_distr.mu.shape[0]).to(self.flags.device)

        # Sample z_0
        z = [in_distr.reparameterize()]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            # z' = z + u h( w^T z + b)
            if self.flags.amortized_flow:
                z_k, log_det_jacobian = flow_k(z[k], flow_params.u[:, k, :, :], flow_params.w[:, k, :, :],
                                               flow_params.b[:, k, :, :])
            else:
                z_k, log_det_jacobian = flow_k(z[k], self.u[:, k, :, :].repeat(num_samples, 1, 1),
                                               self.w[:, k, :, :].repeat(num_samples, 1, 1),
                                               self.b[:, k, :, :].repeat(num_samples, 1, 1))
            z.append(z_k)
            log_det_j += log_det_jacobian

        return z[0], z[-1], log_det_j

    def get_flow_params(self, h):
        # get amortized u an w for all flows
        if self.num_flows and self.flags.amortized_flow:
            return PlanarFlowParams(**{'u': self.amor_u(h).view(h.shape[0], self.num_flows, self.flags.class_dim, 1),
                                       'w': self.amor_w(h).view(h.shape[0], self.num_flows, 1, self.flags.class_dim),
                                       'b': self.amor_b(h).view(h.shape[0], self.num_flows, 1, 1), })
        else:
            return PlanarFlowParams(**{k: None for k in ['u', 'w', 'b']})
