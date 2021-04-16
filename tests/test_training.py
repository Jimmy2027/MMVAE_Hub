import tempfile
from pathlib import Path

import torch
from mmvae_base.modalities.text.alphabet import alphabet
from mmvae_base.utils.filehandling import create_dir_structure

import mmvae_mst
from mmvae_mst import MNISTSVHNText, run_epochs
from mmvae_mst.flags import flags_set_alpha_modalities, setup_flags, parser


def test_run_epochs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        flags = parser.parse_args([])
        flags.config_path = Path(mmvae_mst.__file__).parent.parent / 'configs/toy_config.json'
        flags = setup_flags(flags, testing=True)
        use_cuda = torch.cuda.is_available()
        flags.device = torch.device('cuda' if use_cuda else 'cpu')
        flags.dir_experiment = tmpdirname
        flags = flags_set_alpha_modalities(flags)
        flags.alphabet = alphabet
        flags = create_dir_structure(flags)

        # mst = MNISTSVHNText(flags, alphabet)
        # mst.set_optimizer()
        # run_epochs(mst)


if __name__ == '__main__':
    pass
    # test_run_epochs()
