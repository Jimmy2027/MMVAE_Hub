import json
from pathlib import Path

import optuna
from mmvae_hub.leomed_utils.boilerplate import compress_experiment_run_dir

from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import parser, FlagsSetup
from mmvae_hub.utils.setup.flags_utils import get_config_path


class HyperoptTrainer:
    def __init__(self, flags):
        self.flags = flags

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

        self.flags.optuna = trial
        self.flags.norby = False
        self.flags.use_db = False

        # self.flags.end_epoch = 100
        self.flags.end_epoch = 1
        self.flags.calc_prd = False

        # do this to store values such that they can be retrieved in the database
        self.flags.str_experiment = trial.suggest_categorical('exp_uid', [self.flags.str_experiment])

        self.flags.initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-1, log=True)
        self.flags.class_dim = trial.suggest_categorical("class_dim", [64, 128, 256, 512])
        self.flags.num_flows = trial.suggest_int("num_flows", low=0, high=20, step=1)
        self.flags.beta = trial.suggest_float("beta", 0.01, 2.0)
        mst = PolymnistExperiment(self.flags)
        mst.set_optimizer()

        return run_hyperopt_epochs(PolymnistTrainer(mst))


def run_hyperopt_epochs(trainer: PolymnistTrainer) -> int:
    test_results = trainer.run_epochs()

    if trainer.flags.leomed:
        compress_experiment_run_dir(flags)

    return test_results.hyperopt_score


if __name__ == '__main__':
    flags = parser.parse_args()

    storage_sqlite = optuna.storages.RDBStorage("sqlite:///hyperopt.db", heartbeat_interval=1)
    study = optuna.create_study(direction="maximize", storage=storage_sqlite,
                                study_name=f"distributed-hyperopt-{flags.method}")

    # postgresql_storage_address = "postgresql://klugh@localhost/distributed_hyperopt"
    #
    # study_name = f'distributed-{flags.method}'
    # try:
    #     study = optuna.load_study(study_name=study_name,
    #                               storage=postgresql_storage_address)
    # except:
    #     study = optuna.create_study(direction="maximize", storage=postgresql_storage_address,
    #                                 study_name=study_name)

    flags.dir_experiment = Path(flags.dir_experiment) / 'optuna'
    flags_setup = FlagsSetup(get_config_path(flags))
    flags = flags_setup.setup(flags)
    trainer = HyperoptTrainer(flags)
    study.optimize(trainer.hyperopt, n_trials=1, gc_after_trial=True)
    print("Best trial:")
    print(study.best_params)
    with open('hyperopt_best_results.json', 'w') as jsonfile:
        json.dump(study.best_params, jsonfile)
