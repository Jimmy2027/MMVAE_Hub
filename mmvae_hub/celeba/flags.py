from pathlib import Path

from mmvae_hub.utils.setup.flags_utils import BaseFlagsSetup

from mmvae_hub.base.BaseFlags import parser as parser

# DATASET NAME
parser.add_argument('--name', type=str, default='celeba', help="name of the dataset")
parser.add_argument('--exp_str_prefix', type=str, default='celeba', help="prefix of the experiment directory.")

# add arguments
parser.add_argument('--len_sequence', type=int, default=256, help="length of sequence")
parser.add_argument('--img_size', type=int, default=64, help="img dimension (width/height)")
parser.add_argument('--image_channels', type=int, default=3, help="number of channels in images")
parser.add_argument('--crop_size_img', type=int, default=148, help="number of channels in images")
parser.add_argument('--dir_text', type=str, default='../text', help="directory where text is stored")
parser.add_argument('--random_text_ordering', type=bool, default=False,
                    help="flag to indicate if attributes are shuffled randomly")
parser.add_argument('--random_text_startindex', type=bool, default=True,
                    help="flag to indicate if start index is random")

parser.add_argument('--DIM_text', type=int, default=128, help="filter dimensions of residual layers")
parser.add_argument('--DIM_img', type=int, default=128, help="filter dimensions of residual layers")
parser.add_argument('--num_layers_text', type=int, default=7, help="number of residual layers")
parser.add_argument('--num_layers_img', type=int, default=5, help="number of residual layers")
parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='categorical', help="output distribution")

# classifier
parser.add_argument('--encoder_save_m1', type=str, default='encoderM1', help="model save for encoder")
parser.add_argument('--encoder_save_m2', type=str, default='encoderM2', help="model save for encoder")
parser.add_argument('--decoder_save_m1', type=str, default='decoderM1', help="model save for decoder")
parser.add_argument('--decoder_save_m2', type=str, default='decoderM2', help="model save for decoder")
parser.add_argument('--clf_save_m1', type=str, default='clf_m1', help="model save for clf")
parser.add_argument('--clf_save_m2', type=str, default='clf_m2', help="model save for clf")

# weighting of loss terms
parser.add_argument('--div_weight_m1_content', type=float, default=0.35,
                    help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.35,
                    help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.3,
                    help="default weight divergence term prior")


class CelebaFlagsSetup(BaseFlagsSetup):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.parser = parser

    def flags_set_alpha_modalities(self, flags):
        flags.alpha_modalities = [flags.div_weight_uniform_content, flags.div_weight_m1_content,
                                  flags.div_weight_m2_content]
        return flags
