from pathlib import Path

from mmvae_hub.base.BaseFlags import parser as parser
from mmvae_hub.utils.setup.flags_utils import str2bool, BaseFlagsSetup

parser.add_argument('--exp_str_prefix', type=str, default='Mimic', help="prefix of the experiment directory.")
parser.add_argument('--mods', type=str, default='F_L_T', help="First letter of modalities.")

# Image dependent
parser.add_argument('--fixed_image_extractor', type=str2bool, default=True,
                    help="If the feature extraction layers of the "
                         "pretrained densenet are frozen. "
                         "Only works when img_clf_type classifier "
                         "is densenet.")
# DATA DEPENDENT
parser.add_argument('--only_text_modality', type=str2bool, default=False,
                    help="flag to indicate if only the text modality is to be used")
parser.add_argument('--undersample_dataset', type=str2bool, default=False,
                    help="flag to indicate if the dataset should be undersampled such that there are "
                         "the same number of datapoints that have no label than datapoints that have a label")
parser.add_argument('--weighted_sampler', type=str2bool, default=False,
                    help="If a weighted sampler should be used for the dataloader.")
parser.add_argument('--binary_labels', type=str2bool, default=False,
                    help="If True, label 'Finding' with classes 0 and 1 will be used for the classification evaluation.")

# Text Dependent
parser.add_argument('--len_sequence', type=int, default=128, help="length of sequence")
parser.add_argument('--word_min_occ', type=int, default=3,
                    help="min occurence of a word in the dataset such that it is added to the vocabulary.")
parser.add_argument('--text_gen_lastlayer', type=str, default='softmax',
                    help="Last layer of the text generator. Chose between none, softmax and sigmoid.")

parser.add_argument('--style_pa_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_lat_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_text_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--image_channels', type=int, default=1, help="number of classes on which the data set trained")
parser.add_argument('--img_size', type=int, default=128, help="size of the images on which the model is trained")
parser.add_argument('--DIM_img', type=int, default=128, help="number of classes on which the data set trained")
parser.add_argument('--DIM_text', type=int, default=128, help="number of classes on which the data set trained")
parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m3', type=str, default='categorical', help="output distribution")

# paths to save models
parser.add_argument('--encoder_save_m1', type=str, default='encoderM1', help="model save for encoder")
parser.add_argument('--encoder_save_m2', type=str, default='encoderM2', help="model save for encoder")
parser.add_argument('--encoder_save_m3', type=str, default='encoderM3', help="model save for decoder")
parser.add_argument('--decoder_save_m1', type=str, default='decoderM1', help="model save for decoder")
parser.add_argument('--decoder_save_m2', type=str, default='decoderM2', help="model save for decoder")
parser.add_argument('--decoder_save_m3', type=str, default='decoderM3', help="model save for decoder")

# classifiers
parser.add_argument('--text_clf_type', type=str, default='word',
                    help="text classifier type, implemented are 'word' and 'char'")
parser.add_argument('--img_clf_type', type=str, default='resnet',
                    help="image classifier type, implemented are 'resnet' and 'densenet'")
parser.add_argument('--clf_save_m1', type=str, default='clf_m1', help="model save for clf")
parser.add_argument('--clf_save_m2', type=str, default='clf_m2', help="model save for clf")
parser.add_argument('--clf_save_m3', type=str, default='clf_m3', help="model save for clf")
parser.add_argument('--clf_loss', type=str, default='binary_crossentropy',
                    choices=['binary_crossentropy', 'crossentropy', 'bce_with_logits'], help="model save for clf")

# Callbacks
parser.add_argument('--reduce_lr_on_plateau', type=bool, default=False,
                    help="boolean indicating if callback 'reduce lr on plateau' is used")
parser.add_argument('--max_early_stopping_index', type=int, default=5,
                    help="patience of the early stopper. If the target metric did not improve "
                         "for that amount of epochs, training is stopepd")
parser.add_argument('--start_early_stopping_epoch', type=int, default=0,
                    help="epoch on which to start the early stopping callback")

# LOSS TERM WEIGHTS
parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--beta_m3_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--div_weight_m1_content', type=float, default=0.25,
                    help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.25,
                    help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_m3_content', type=float, default=0.25,
                    help="default weight divergence term content modality 3")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.25,
                    help="default weight divergence term prior")
parser.add_argument('--rec_weight_m1', default=0.33, type=float,
                    help="weight of the m1 modality for the log probs. Type should be either float or string.")
parser.add_argument('--rec_weight_m2', default=0.33, type=float,
                    help="weight of the m2 modality for the log probs. Type should be either float or string.")
parser.add_argument('--rec_weight_m3', default=0.33, type=float,
                    help="weight of the m3 modality for the log probs. Type should be either float or string.")


class MimicFlagsSetup(BaseFlagsSetup):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.parser = parser

    def flags_set_alpha_modalities(self, flags):
        flags.alpha_modalities = [flags.div_weight_uniform_content, flags.div_weight_m1_content,
                                  flags.div_weight_m2_content, flags.div_weight_m3_content]
        return flags
