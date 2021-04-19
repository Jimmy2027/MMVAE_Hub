from mmvae_hub.base.utils.flags_utils import get_config_path
from mmvae_hub.mmnist import MmnistTrainer
from mmvae_hub.mmnist.experiment import MMNISTExperiment
from mmvae_hub.mmnist.flags import FlagsSetup, parser

if __name__ == '__main__':
    flags = parser.parse_args()
    flags_setup = FlagsSetup(get_config_path(flags))
    flags = flags_setup.setup(flags)
    mst = MMNISTExperiment(flags)
    mst.set_optimizer()
    trainer = MmnistTrainer(mst)
    trainer.run_epochs()
