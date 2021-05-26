from mmvae_hub import log
from mmvae_hub.base.BaseFlags import parser as parser
from mmvae_hub.utils import get_method
from mmvae_hub.utils import get_freer_gpu, update_flags_with_config

# DATASET NAME
parser.add_argument('--dataset', type=str, default='SVHN_MNIST_text', help="name of the dataset")
parser.add_argument('--exp_str_prefix', type=str, default='MST', help="prefix of the experiment directory.")
# DATA DEPENDENT
# to be set by experiments themselves
parser.add_argument('--style_m1_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_m2_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_m3_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--len_sequence', type=int, default=8, help="length of sequence")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes on which the data set trained")
parser.add_argument('--dim', type=int, default=64, help="number of classes on which the data set trained")
parser.add_argument('--data_multiplications', type=int, default=20, help="number of pairs per sample")
parser.add_argument('--num_hidden_layers', type=int, default=1, help="number of channels in images")
parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m3', type=str, default='categorical', help="output distribution")

# SAVE and LOAD
# to bet set by experiments themselves
parser.add_argument('--encoder_save_m1', type=str, default='encoderM1', help="model save for encoder")
parser.add_argument('--encoder_save_m2', type=str, default='encoderM2', help="model save for encoder")
parser.add_argument('--encoder_save_m3', type=str, default='encoderM3', help="model save for decoder")
parser.add_argument('--decoder_save_m1', type=str, default='decoderM1', help="model save for decoder")
parser.add_argument('--decoder_save_m2', type=str, default='decoderM2', help="model save for decoder")
parser.add_argument('--decoder_save_m3', type=str, default='decoderM3', help="model save for decoder")
parser.add_argument('--clf_save_m1', type=str, default='clf_m1', help="model save for clf")
parser.add_argument('--clf_save_m2', type=str, default='clf_m2', help="model save for clf")
parser.add_argument('--clf_save_m3', type=str, default='clf_m3', help="model save for clf")

# LOSS TERM WEIGHTS
parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--beta_m3_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--div_weight_m1_content', type=float, default=0.25,
                    help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.25,
                    help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_m3_content', type=float, default=0.25,
                    help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.25,
                    help="default weight divergence term prior")


def setup_flags(flags, testing=False):
    """
    If testing is true, no cli arguments will be read.
    """
    import torch
    from pathlib import Path
    import numpy as np
    if flags.config_path:
        flags = update_flags_with_config(config_path=flags.config_path, testing=testing)
    # flags = expand_paths(flags)
    use_cuda = torch.cuda.is_available()
    flags.device = torch.device('cuda' if use_cuda else 'cpu')
    if str(flags.device) == 'cuda':
        torch.cuda.set_device(get_freer_gpu())
    flags = flags_set_alpha_modalities(flags)
    flags.log_file = log.manager.root.handlers[1].baseFilename
    flags.len_sequence = 128 if flags.text_encoding == 'word' else 1024

    if flags.load_flags:
        old_flags = torch.load(Path(flags.load_flags).expanduser())
        # create param dict from all the params of old_flags that are not paths
        params = {k: v for k, v in old_flags.item() if ('dir' not in v) and ('path' not in v)}
        flags.__dict__.update(params)

    if not flags.seed:
        # set a random seed
        flags.seed = np.random.randint(0, 10000)
    flags = get_method(flags)
    return flags


def flags_set_alpha_modalities(flags):
    flags.alpha_modalities = [flags.div_weight_uniform_content, flags.div_weight_m1_content,
                              flags.div_weight_m2_content, flags.div_weight_m3_content]
    return flags
