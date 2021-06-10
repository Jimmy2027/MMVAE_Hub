import sys
import os
import json

import torch

from run_epochs import run_epochs

from mmvae_hub.utils.setup.filehandling import create_dir_structure
from mmvae_hub.utils.setup.filehandling import create_dir_structure_testing
from mmvae_hub.mnistsvhntext.flags import parser
from mmvae_hub.mnistsvhntext.experiment import MNISTSVHNText

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    use_cuda = torch.cuda.is_available();
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu');

    if FLAGS.method == 'poe':
        FLAGS.modality_poe=True;
    elif FLAGS.method == 'moe':
        FLAGS.modality_moe=True;
    elif FLAGS.method == 'jsd':
        FLAGS.modality_jsd=True;
    elif FLAGS.method == 'joint_elbo':
        FLAGS.joint_elbo=True;
    else:
        print('method implemented...exit!')
        sys.exit();
    print(FLAGS.modality_poe)
    print(FLAGS.modality_moe)
    print(FLAGS.modality_jsd)
    print(FLAGS.joint_elbo)

    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content]

    FLAGS = create_dir_structure(FLAGS)
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json');
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    mst = MNISTSVHNText(FLAGS, alphabet);
    create_dir_structure_testing(mst);
    mst.set_optimizer();

    run_epochs(mst);
