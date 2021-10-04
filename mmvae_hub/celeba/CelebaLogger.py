from mmvae_hub.utils.BaseTBLogger import BaseTBLogger


class CelebALogger(BaseTBLogger):
    def __init__(self, name, writer):
        super().__init__(name, writer)

    def write_lr_eval(self, lr_eval: dict):
        """
        write lr eval results to tensorboard logger.
        """
        for l_key, l_val in lr_eval.items():
            if l_val is not None:
                for s, s_key in enumerate(sorted(l_val.keys())):
                    self.writer.add_scalars(f'Latent_Representation_{l_key}/{s_key}',
                                            {'avg_prec': l_val[s_key]['avg_prec']}, self.step)
