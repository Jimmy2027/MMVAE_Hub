from pathlib import Path

from mmvae_hub.base.BaseFlags import parser as parser
from mmvae_hub.utils.setup.flags_utils import BaseFlagsSetup

parser.add_argument('--name', type=str, default='polymnist', help="name of the dataset")
parser.add_argument('--dataset', type=str, default='polymnist', help="name of the dataset")
parser.add_argument('--exp_str_prefix', type=str, default='polymnist', help="prefix of the experiment directory.")
# training

parser.add_argument('--num_mods', type=int, default=3, help="dimension of varying factor latent space")
parser.add_argument('--style_dim', type=int, default=0,
                    help="style dimensionality")  # TODO: use modality-specific style dimensions?
# parser.add_argument('--style_m1_dim', type=int, default=0, help="dimension of varying factor latent space")
# parser.add_argument('--style_m2_dim', type=int, default=0, help="dimension of varying factor latent space")
# parser.add_argument('--style_m3_dim', type=int, default=0, help="dimension of varying factor latent space")

parser.add_argument('--num_classes', type=int, default=10, help="number of classes on which the data set trained")

parser.add_argument('--len_sequence', type=int, default=8, help="length of sequence")
parser.add_argument('--img_size_m1', type=int, default=28, help="img dimension (width/height)")
parser.add_argument('--num_channels_m1', type=int, default=1, help="number of channels in images")
parser.add_argument('--img_size_m2', type=int, default=32, help="img dimension (width/height)")
parser.add_argument('--num_channels_m2', type=int, default=3, help="number of channels in images")
parser.add_argument('--dim', type=int, default=64, help="number of classes on which the data set trained")
parser.add_argument('--data_multiplications', type=int, default=1, help="number of pairs per sample")
parser.add_argument('--num_hidden_layers', type=int, default=1, help="number of channels in images")

# multimodal

parser.add_argument('--subsampled_reconstruction', default=True, help="subsample reconstruction path")
parser.add_argument('--include_prior_expert', action='store_true', default=False, help="factorized_representation")

# weighting of loss terms
parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--beta_m3_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--div_weight', type=float, default=None,
                    help="default weight divergence per modality, if None use 1/(num_mods+1).")
parser.add_argument('--div_weight_uniform_content', type=float, default=None,
                    help="default weight divergence term prior, if None use (1/num_mods+1)")


class FlagsSetup(BaseFlagsSetup):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.parser = parser

    def flags_set_alpha_modalities(self, flags):
        flags.alpha_modalities = [flags.div_weight_uniform_content]
        if flags.div_weight is None:
            flags.div_weight = 1 / (flags.num_mods + 1)
        flags.alpha_modalities.extend([flags.div_weight for _ in range(flags.num_mods)])
        return flags
