import tempfile

import pytest
import torch

from mmvae_hub.networks.FlowVaes import PlanarMixtureMMVae
from mmvae_hub.networks.MixtureVaes import MOEMMVae
from mmvae_hub.networks.utils.mixture_component_selection import mixture_component_selection
from tests.utils import set_me_up


@pytest.mark.tox
def test_fuse_modalities_1():
    """
    With 3 experts and a batch size of 1, the mixture selection should select one of the experts randomly.
    """
    class_dim = 3
    batch_size = 1
    num_mods = 3
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname, method='planar_mixture',
                        attributes={'num_mods': num_mods, 'class_dim': class_dim, 'device': 'cpu',
                                    'batch_size': batch_size, 'weighted_mixture': False})

        model: PlanarMixtureMMVae = mst.mm_vae

        enc_mods: Mapping[str, EncModPlanarMixture] = {
            mod_str: EncModPlanarMixture(None, None, z0=None,
                                         zk=torch.ones((batch_size, class_dim)) * numbr) for numbr, mod_str
            in zip(range(num_mods), mst.modalities)}

        joint_zk = model.mixture_component_selection(enc_mods, 'm0_m1_m2', weight_joint=False)
        assert torch.all(joint_zk == Tensor([[1., 1., 1.]]))


@pytest.mark.tox
def test_fuse_modalities_2():
    """
    With 3 experts and a batch size of 3, the mixture selection should select each of the experts one time.
    """
    class_dim = 3
    batch_size = 3
    num_mods = 3
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname, method='planar_mixture',
                        attributes={'num_mods': num_mods, 'class_dim': class_dim, 'device': 'cpu',
                                    'batch_size': batch_size, 'weighted_mixture': False})

        model: PlanarMixtureMMVae = mst.mm_vae

        enc_mods: Mapping[str, EncModPlanarMixture] = {
            mod_str: EncModPlanarMixture(None, None, z0=None,
                                         zk=torch.ones((batch_size, class_dim)) * numbr) for numbr, mod_str
            in zip(range(num_mods), mst.modalities)}

        joint_zk = model.mixture_component_selection(enc_mods, 'm0_m1_m2', weight_joint=False)
        assert torch.all(joint_zk ==
                         Tensor([[0., 0., 0.],
                                 [1., 1., 1.],
                                 [2., 2., 2.]]))


@pytest.mark.tox
def test_fuse_modalities_3():
    """
    test that during the mixture selection, the batches don't get mixed up.
    If 3 experts have a zk of:
    Tensor([[0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.]])

    (of shape (batch_size, class dim)).

    Then the mixture selection should select the first batch from expert 1, the second batch from expert 2 and the
    third batch from expert 3, giving the same result.
    """
    class_dim = 3
    batch_size = 3
    num_mods = 3
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname, method='planar_mixture',
                        attributes={'num_mods': num_mods, 'class_dim': class_dim, 'device': 'cpu',
                                    'batch_size': batch_size, 'weighted_mixture': False})

        model: PlanarMixtureMMVae = mst.mm_vae

        enc_mod_zk = torch.ones((batch_size, class_dim))
        enc_mod_zk[0] = enc_mod_zk[0] * 0
        enc_mod_zk[2] = enc_mod_zk[2] * 2

        enc_mods: Mapping[str, EncModPlanarMixture] = {
            mod_str: EncModPlanarMixture(None, None, z0=None, zk=enc_mod_zk) for mod_str in mst.modalities}

        joint_zk = model.mixture_component_selection(enc_mods, 'm0_m1_m2', weight_joint=False)
        assert torch.all(joint_zk ==
                         Tensor([[0., 0., 0.],
                                 [1., 1., 1.],
                                 [2., 2., 2.]]))


@pytest.mark.tox
def test_fuse_modalities_4():
    """
    With 3 experts and a batch size of 3, the mixture selection should select each of the experts one time.
    """
    class_dim = 3
    batch_size = 3
    num_mods = 3
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname, method='moe',
                        attributes={'num_mods': num_mods, 'class_dim': class_dim, 'device': 'cpu',
                                    'batch_size': batch_size, 'weighted_mixture': False})

        model: MOEMMVae = mst.mm_vae

        mus = torch.ones((num_mods, batch_size, class_dim))
        mus[0] = mus[0] * 0
        mus[2] = mus[2] * 2

        logvars = torch.zeros((num_mods, batch_size, class_dim))

        w_modalities = torch.ones((num_mods,)) * (1/3)

        joint_distr = mixture_component_selection(mst.flags, mus, logvars, w_modalities)
        assert torch.all(joint_distr.mu ==
                         Tensor([[0., 0., 0.],
                                 [1., 1., 1.],
                                 [2., 2., 2.]]))


if __name__ == '__main__':
    test_fuse_modalities_4()
