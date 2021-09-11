from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.mnistsvhntext.experiment import MNISTSVHNText
from mmvae_hub.mnistsvhntext.mnistsvhntextTrainer import mnistsvhnTrainer
from mmvae_hub.mnistsvhntext.flags import mnistsvhntextFlagsSetup, parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

search_space1 = {
    'method': 'mopoe',
    "initial_learning_rate": 0.0005,
    'class_dim': 640,
    "min_beta": 0,
    "max_beta": 2.0,
    "beta_warmup": 50,
    "num_gfm_flows": 3,
    "num_mods": 3,
    "end_epoch": 150,
    "eval_freq": 10,
}

if __name__ == '__main__':

    for grid in [search_space1]:
        # for sp in ParameterGrid(grid):
        for sp in [grid]:
            # for _ in [1]:
            flags = parser.parse_args()
            flags_setup = mnistsvhntextFlagsSetup(get_config_path(dataset='mnistsvhntext', flags=flags))
            flags = flags_setup.setup(flags, additional_args=sp)
            with norby(f'Starting Experiment {flags.experiment_uid}.', f'Experiment {flags.experiment_uid} finished.'):
                mst = MNISTSVHNText(flags)
                mst.set_optimizer()
                trainer = mnistsvhnTrainer(mst)
                trainer.run_epochs()
