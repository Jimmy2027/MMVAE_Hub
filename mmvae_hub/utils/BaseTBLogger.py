class BaseTBLogger:
    def __init__(self, name, writer):
        self.name = name
        self.writer = writer
        self.training_prefix = 'train'
        self.testing_prefix = 'test'
        self.step = 0

    def write_log_probs(self, name, log_probs):
        self.writer.add_scalars('%s/LogProb' % name,
                                log_probs,
                                self.step)

    def write_klds(self, name, klds):
        self.writer.add_scalars('%s/KLD' % name,
                                klds,
                                self.step)

    def write_group_div(self, name, group_div):
        self.writer.add_scalars('%s/group_divergence' % name,
                                {'group_div': group_div},
                                self.step)

    def write_latent_distr(self, name, enc_mods: dict):
        for k, key in enumerate(enc_mods.keys()):
            # if enc_mods[key]['latents_class']['mu'] is not None:
            self.writer.add_scalars('%s/mu' % name,
                                    {key: enc_mods[key]['latents_class']['mu']},
                                    self.step)
            # if enc_mods[key]['latents_class']['logvar'] is not None:
            self.writer.add_scalars('%s/logvar' % name,
                                    {key: enc_mods[key]['latents_class']['logvar']},
                                    self.step)

    def write_lr_eval(self, lr_eval: dict):
        """
        write lr eval results to tensorboard logger.
        """
        for l_key, l_val in lr_eval.items():
            if l_val is not None:
                for s, s_key in enumerate(sorted(l_val.keys())):
                    self.writer.add_scalars(f'Latent_Representation_{l_key}/{s_key}',
                                            {'accuracy': l_val[s_key]['accuracy']}, self.step)

    def write_coherence_logs(self, gen_eval):
        for j, l_key in enumerate(sorted(gen_eval['cond'].keys())):
            for k, s_key in enumerate(gen_eval['cond'][l_key].keys()):
                self.writer.add_scalars('Generation/%s/%s' %
                                        (l_key, s_key),
                                        gen_eval['cond'][l_key][s_key],
                                        self.step)
        self.writer.add_scalars('Generation/Random', gen_eval['random'], self.step)

    def write_lhood_logs(self, lhoods):
        for k, key in enumerate(sorted(lhoods.keys())):
            self.writer.add_scalars('Likelihoods/%s' % (key), lhoods[key], self.step)

    def write_prd_scores(self, prd_scores):
        self.writer.add_scalars('PRD',
                                prd_scores,
                                self.step)

    def write_plots(self, plots, epoch):
        for p_key, ps in plots.items():
            for name, fig in ps.items():
                self.writer.add_image(p_key + '_' + name, fig, epoch, dataformats="HWC")

    def add_basic_logs(self, name, joint_divergence, loss, log_probs, klds):
        self.writer.add_scalars('%s/Loss' % name,
                                {'loss': loss},
                                self.step)
        self.write_log_probs(name, log_probs)
        self.write_klds(name, klds)
        self.write_group_div(name, joint_divergence)
        # self.write_latent_distr(name, enc_mods=latents)

    def write_training_logs(self, joint_divergence, total_loss, log_probs, klds):
        self.add_basic_logs(self.training_prefix, joint_divergence, total_loss, log_probs,
                            klds)

    def write_testing_logs(self, joint_divergence, total_loss, log_probs, klds):
        self.add_basic_logs(self.testing_prefix, joint_divergence, total_loss, log_probs, klds)
