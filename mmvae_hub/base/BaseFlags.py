import argparse

from mmvae_hub.utils.setup.flags_utils import str2bool

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help="Which dataset to use.")
parser.add_argument('--experiment_uid', type=str, help="Unique identifier of experiment run.", default=None)
parser.add_argument('--config_path', type=str, default=None, help="Path to the json config.")
parser.add_argument('--leomed', type=str2bool, default=False,
                    help="If experiment is running on the leomed cluster, set this flag to True. This will make sure "
                         "to set the temporary directories to TMPDIR.")
parser.add_argument('--norby', type=str2bool, default=False,
                    help="If true use norby package to send training updates via telegram. "
                         "(Needs norby to be configured on system)")
parser.add_argument('--use_db', default=False, action="store_true",
                    help="If set, will send experiment results to a database for further analysis. "
                         "The configuration file containing the URI to connect to the db needs to be"
                         " under configs/mmvaedb_config.py. If this flag is set to True and the db is unreachable, "
                         "the results will be stored such that they can then be easily uploaded to the db.")

parser.add_argument('--use_cuda', type=bool, default=True, help="Bool to indicate if GPU should be used.")
parser.add_argument('--deterministic', type=bool, default=False,
                    help="Bool to indicate if experiment should be run in a deterministic manner to produce "
                         "reproducible results. Does not work with cuda enabled, so experiment will run on "
                         "CPU if set to True.")
parser.add_argument('--optuna', type=str2bool, default=False,
                    help="Indicates if optuna's hyperoptimization is used.")

# TRAINING
parser.add_argument('--seed', type=int, default=None,
                    help="Random seed for reproducibility. If None, will be set randomly.")
parser.add_argument('--distributed', type=bool, default=False,
                    help="flag to indicate if torch.nn.DataParallel is used")
parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")
parser.add_argument('--steps_per_training_epoch', type=int, default=None,
                    help="flag to indicate the number of steps in a training epoch.")

# DATA DEPENDENT
parser.add_argument('--class_dim', type=int, default=20, help="dimension of common factor latent space")
parser.add_argument('--dataloader_workers', type=int, default=8, help="number of workers used for the Dataloader")

# SAVE and LOAD
parser.add_argument('--mm_vae_save', type=str, default='mm_vae', help="model save for vae_bimodal")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--load_flags', type=str, default=None, help="overwrite all values with parameters from an old "
                                                                 "experiment. Give the path to the flags.rar "
                                                                 "file as input.")

# DIRECTORIES
# clfs
parser.add_argument('--dir_clf', type=str, default='../clf', help="directory where clf is stored")
# data
parser.add_argument('--dir_data', type=str, default='../data', help="directory where data is stored")
# experiments
parser.add_argument('--dir_experiment', type=str, default='/tmp/multilevel_multimodal_vae_swapping',
                    help="directory to save generated samples in")
# fid
parser.add_argument('--dir_fid', type=str, default=None,
                    help="directory to save generated samples for fid score calculation")
# fid_score
parser.add_argument('--inception_state_dict', type=str, default='../inception_state_dict.pth',
                    help="path to inception v3 state dict")

# EVALUATION
parser.add_argument('--use_clf', default=False, action="store_true",
                    help="flag to indicate if generates samples should be classified")
parser.add_argument('--calc_nll', type=str2bool, default=True,
                    help="flag to indicate calculation of nll")
parser.add_argument('--eval_lr', default=False, action="store_true",
                    help="flag to indicate evaluation of latent representations")
parser.add_argument('--calc_prd', default=False, action="store_true",
                    help="flag to indicate calculation of prec-rec for gen model")
parser.add_argument('--save_figure', default=False, action="store_true",
                    help="flag to indicate if figures should be saved to disk (in addition to tensorboard logs). "
                         "Is set to true if calc_prd is true.")
parser.add_argument('--eval_freq', type=int, default=10,
                    help="frequency of evaluation of latent representation of generative performance (in number of epochs)")
parser.add_argument('--eval_freq_fid', type=int, default=10,
                    help="frequency of evaluation of latent representation of generative performance (in number of epochs)")
parser.add_argument('--num_samples_fid', type=int, default=10000,
                    help="number of samples the calculation of fid is based on")
parser.add_argument('--num_training_samples_lr', type=int, default=1000,
                    help="number of training samples to train the lr clf")

# multimodal
parser.add_argument('--method', type=str, default='poe', help='choose method for training the model')
parser.add_argument('--modality_jsd', type=bool, default=False, help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False, help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False, help="modality_moe")
parser.add_argument('--joint_elbo', type=bool, default=False, help="modality_moe")
parser.add_argument('--poe_unimodal_elbos', type=bool, default=False, help="unimodal_klds")
parser.add_argument('--factorized_representation', action='store_true', default=False,
                    help="factorized_representation")
parser.add_argument('--weighted_mixture', type=str2bool, default=False,
                    help="Flag that indicates if the experts are selected randomly or with a probability that is "
                         "weighted by the inverse of the variance of the expert during the mixture.")
parser.add_argument('--feature_extractor_img', type=str, default='resnet', help='which feature extractor model to use',
                    choices=['resnet', 'densenet'])

# LOSS TERM WEIGHTS
parser.add_argument('--beta', type=float, default=0, help="default weight of sum of weighted divergence terms")
parser.add_argument('--prior', type=str, default='laplace', help="prior used to compute the KL divergence.")
parser.add_argument('--qz_x', type=str, default='laplace', help="distribution used for the uni modal posteriors.")
parser.add_argument('--beta_style', type=float, default=1.0,
                    help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0,
                    help="default weight of sum of weighted content divergence terms")

# kl annealing
parser.add_argument('--beta_warmup', type=int, default=100, metavar='N',
                    help='number of epochs for warm-up. Set to 0 to turn warmup off.')
parser.add_argument('--max_beta', type=float, default=1., help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, help='min beta for warm-up')
parser.add_argument('--beta_start_epoch', type=float, default=0.0, help='epoch at which beta starts increasing')

# FLOWS
parser.add_argument('--num_flows', type=int, default=4, help="Number of flow layers, ignored in absence of flows.")
parser.add_argument('--amortized_flow', type=str2bool, default=False,
                    help="If True, use amortized flows, as described in Berg et al. 2019")
parser.add_argument('--coupling_dim', type=int, default=512, help="Dimensions of the coupling layers in flow.")

# Generalized f-Means
parser.add_argument('--num_gfm_flows', type=int, default=4,
                    help="Number of flow layers that are used to implement the GfM function.")
parser.add_argument('--nbr_coupling_block_layers', type=int, default=2,
                    help="Number of layers used in a coupling block, additional to the input and output layer.")

# importance sampling
parser.add_argument('--K', type=int, default=10,
                    help="Number of flow layers that are used to implement the GfM function.")
