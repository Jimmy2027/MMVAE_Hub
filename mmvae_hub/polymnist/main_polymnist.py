from sklearn.model_selection import ParameterGrid

from mmvae_hub.base.utils.flags_utils import get_config_path
from mmvae_hub.polymnist import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser

search_spaces = {
    'method': ['planar_mixture'],
    # 'method': ['moe'],
    'class_dim': [512],
    "beta": [2.5],
    "num_flows": [0],
    "num_mods": [1]
}

search_spaces_1 = {
    'method': ['planar_mixture'],
    # 'method': ['moe'],
    'class_dim': [512],
    "beta": [2.5],
    "num_flows": [5],
    "num_mods": [3]
}

if __name__ == '__main__':
    for grid in [search_spaces_1]:
        for sp in ParameterGrid(grid):
            # for _ in [1]:
            flags = parser.parse_args()
            flags_setup = FlagsSetup(get_config_path(flags))
            flags = flags_setup.setup(flags, additional_args=sp)
            # flags = flags_setup.setup(flags)
            mst = PolymnistExperiment(flags)
            mst.set_optimizer()
            trainer = PolymnistTrainer(mst)
            trainer.run_epochs()
