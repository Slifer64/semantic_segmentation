#! /bin/bash

source bash/utils.sh
source bash/set_datasets.sh


SAVE_TO="data/train_set"
ITERS=20000
SEED=0

python3 augmerge_dataset.py --datasets $all_datasets --items_range $items_range --bg_dataset=data/bg_dataset/ --save_to=$SAVE_TO --iters=$ITERS --seed=$SEED --viz=0
