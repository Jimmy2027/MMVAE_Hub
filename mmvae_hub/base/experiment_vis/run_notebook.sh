#!/usr/bin/env bash

EXPERIMENT_DIR="${1}"

cp experiment_vis.ipynb "$EXPERIMENT_DIR"/ || exit 1

#jupyter nbconvert --to notebook --execute "$EXPERIMENT_DIR"/experiment_vis.ipynb --allow-errors || exit 1
#jupyter nbconvert --to html "$EXPERIMENT_DIR"/experiment_vis.nbconvert.ipynb || exit 1
#jupyter nbconvert --to pdf "$EXPERIMENT_DIR"/experiment_vis.nbconvert.ipynb || exit 1
exp_id="$(basename "${EXPERIMENT_DIR}")"
echo "Experiment ID: $exp_id"
python -c "from ppb import upload; from mmbae_base.base.utils.MongoDB import MongoDatabase; expvis_url = ppb.upload($EXPERIMENT_DIR/experiment_vis.nbconvert.pdf, plain=True); db = MongoDatabase(training=False, _id:$exp_id);print(db); db.insert_dict({'expvis_url': expvis_url})"
