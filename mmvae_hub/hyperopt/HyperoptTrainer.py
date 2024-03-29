import json
import shutil
from pathlib import Path

import optuna

from mmvae_hub import log
from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import parser, FlagsSetup
from mmvae_hub.utils.setup.flags_utils import get_config_path


class HyperoptTrainer:
    def __init__(self, flags, flags_setup: FlagsSetup, dataset: str, method: str):
        self.flags = flags
        self.flags_setup = flags_setup
        self.dataset = dataset
        self.method = method

    def hyperopt(self, trial):
        """
        FLAGS: Namespace = parser.parse_args()
        FLAGS.config_path = get_config_path(FLAGS)

        main = Main(FLAGS)
        study.optimize(main.hyperopt, n_trials=100, gc_after_trial=True, callbacks=[optuna_cb])
        print("Best trial:")
        print(study.best_params)
        with open('hyperopt_best_results.json', 'w') as jsonfile:
            json.dump(study.best_params, jsonfile)
            """
        self.flags = self.flags_setup.setup(self.flags,
                                            additional_args={'dataset': self.dataset, 'method': self.method})

        self.flags.optuna = trial
        self.flags.norby = False
        self.flags.use_db = False

        self.flags.beta_warmup = 30

        self.flags.end_epoch = 100
        # self.flags.end_epoch = 1
        self.flags.calc_prd = True
        self.flags.calc_nll = False

        eval_freq = 100
        self.flags.eval_freq_fid = eval_freq
        self.flags.eval_freq = eval_freq

        # do this to store values such that they can be retrieved in the database
        # self.flags.str_experiment = trial.suggest_categorical('exp_uid', [self.flags.str_experiment])

        # self.flags.initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-3, log=True)
        self.flags.class_dim = trial.suggest_categorical("class_dim", [1280])
        self.flags.beta = trial.suggest_float("max_beta", 1.0, 4)

        self.flags.initial_learning_rate = 0.0005
        # self.flags.class_dim = 640
        # self.flags.beta = 2.0

        if method in ['mopgfm', 'mogfm']:
            self.flags.coupling_dim = trial.suggest_categorical("coupling_dim", [64, 128, 256])
            self.flags.num_flows = trial.suggest_int("num_gfm_flows", low=1, high=15, step=2)
            self.flags.num_flows = trial.suggest_int("nbr_coupling_block_layers", low=0, high=10, step=2)

        mst = PolymnistExperiment(self.flags)
        mst.set_optimizer()

        # try:
        return run_hyperopt_epochs(PolymnistTrainer(mst))
        # except Exception as e:
        #     log.info(f'Experiment failed with: {e}')
        #     return None


def run_hyperopt_epochs(trainer: PolymnistTrainer) -> int:
    test_results = trainer.run_epochs()

    # clean experiment run dir
    shutil.rmtree(trainer.flags.dir_experiment_run)
    log.info(f'Finished hyperopt run with score: {test_results.hyperopt_score}.')
    return test_results.hyperopt_score


if __name__ == '__main__':
    dataset = 'polymnist'
    method = 'mopgfm'
    flags = parser.parse_args()

    study_name = f'hyperopt-{method}2'

    # storage_sqlite = optuna.storages.RDBStorage("sqlite:///hyperopt.db", heartbeat_interval=1)
    # study = optuna.create_study(direction="maximize", storage=storage_sqlite,
    #                             study_name=f"distributed-hyperopt-{flags.method}")

    postgresql_storage_address = "postgresql://klugh@ethsec-login-03:5433/distributed_hyperopt"

    try:
        study = optuna.load_study(study_name=study_name,
                                  storage=postgresql_storage_address)
    except:
        study = optuna.create_study(direction="maximize", storage=postgresql_storage_address,
                                    study_name=study_name)

    flags.dir_experiment = Path(flags.dir_experiment) / 'optuna'
    flags_setup = FlagsSetup(get_config_path(dataset=dataset, flags=flags))
    trainer = HyperoptTrainer(flags, flags_setup, dataset=dataset, method=method)
    study.optimize(trainer.hyperopt, n_trials=100, gc_after_trial=True)
    print("Best trial:")
    print(study.best_params)
    with open('hyperopt_best_results.json', 'w') as jsonfile:
        json.dump(study.best_params, jsonfile)
