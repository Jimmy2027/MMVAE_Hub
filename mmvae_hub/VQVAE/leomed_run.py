from sklearn.model_selection import ParameterGrid

from mmvae_hub.leomed_utils.launch_jobs import launch_leomed_jobs

search_space = {
    'gpu_mem': [12000],
    'n_gpus': [1],
    'method': ['vqmopoe'],
    'mods': ['F_L'],
    # 'mods': ['T'],
    'class_dim': [512],
    "end_epoch": [300],
    "initial_learning_rate": [1e-4],
    # "num_gfm_flows": [3, 5]

}
for params in ParameterGrid(search_space):
    launch_leomed_jobs(which_dataset='mimic', params=params)
