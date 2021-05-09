from sklearn.model_selection import ParameterGrid

from mmvae_hub.base.utils.flags_utils import get_config_path
from mmvae_hub.polymnist import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser

search_spaces = {
    'method': ['moe', 'joint_elbo'],
    'class_dim': [32, 64, 128, 256, 512],
    "beta": [0.5, 1, 2.5, 3, 5]
}

if __name__ == '__main__':
    for sp in ParameterGrid(search_spaces):
        flags = parser.parse_args()
        flags_setup = FlagsSetup(get_config_path(flags))
        flags = flags_setup.setup(flags, additional_args=sp)
        mst = PolymnistExperiment(flags)
        mst.set_optimizer()
        trainer = PolymnistTrainer(mst)
        trainer.run_epochs()
