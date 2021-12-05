from mmvae_hub.utils.BaseTBLogger import BaseTBLogger


class VQTBLogger(BaseTBLogger):
    def __init__(self, name, writer):
        super().__init__(name, writer)

    def write_basic_logs(self, total_loss, quant_losses, rec_losses, phase: str):
        self.writer.add_scalars(f'{phase}/Loss',
                                {'loss': total_loss},
                                self.step)
        self.writer.add_scalars(f'{phase}/quant_losses',
                                quant_losses,
                                self.step)
        for s_key, s_rec_losses in rec_losses.items():
            self.writer.add_scalars(f'{phase}/rec_losses/{s_key}',
                                    s_rec_losses,
                                    self.step)
