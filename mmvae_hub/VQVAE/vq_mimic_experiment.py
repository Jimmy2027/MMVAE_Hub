import os
import typing

from mmvae_hub import log
from mmvae_hub.VQVAE.VQMimicIMG import VQMimicPA, VQMimicLateral
from mmvae_hub.VQVAE.VQMimicText import VQMimicText
from mmvae_hub.mimic.experiment import MimicExperiment
from mmvae_hub.modalities import BaseModality


class VQmimicExperiment(MimicExperiment):
    def __init__(self, flags):
        super().__init__(flags)

    def set_modalities(self) -> typing.Mapping[str, BaseModality]:
        log.info('setting modalities')
        mods = {}
        for mod_str in self.flags.mods.split('_'):
            if mod_str == 'F':
                mod = VQMimicPA(self.flags, self.labels, self.flags.rec_weight_m1, self.plot_img_size)
            elif mod_str == 'L':
                mod = VQMimicLateral(self.flags, self.labels, self.flags.rec_weight_m2, self.plot_img_size)
            elif mod_str == 'T':
                mod = VQMimicText(self.flags, self.labels, self.flags.rec_weight_m3, self.plot_img_size,
                                  self.dataset_train.report_findings_dataset.i2w)
            else:
                raise ValueError(f'Invalid mod_str {mod_str}.' + 'Choose between {F,L,T}')
            mods[mod.name] = mod

        return mods

    def set_paths_fid(self):
        dir_real = os.path.join(self.flags.dir_gen_eval_fid, 'real')
        paths = {'real': dir_real, }
        dir_cond = self.flags.dir_gen_eval_fid
        for name in [*self.subsets]:
            paths[name] = os.path.join(dir_cond, name)
        return paths
