import torch.nn as nn


class LinearFeatureCompressor(nn.Module):
    """
    Calculates mu and logvar from latent representations given by the encoder.
    Independent of modality
    """

    def __init__(self, in_channels, out_channels_style, out_channels_content):
        super(LinearFeatureCompressor, self).__init__()

        self.content_mu = nn.Linear(in_channels, out_channels_content, bias=True)
        self.content_logvar = nn.Linear(in_channels, out_channels_content, bias=True)

    def forward(self, feats):
        feats = feats.view(feats.size(0), -1)
        mu_content, logvar_content = self.content_mu(feats), self.content_logvar(feats)

        return mu_content, logvar_content
