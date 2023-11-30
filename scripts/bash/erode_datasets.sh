#! /bin/bash

source bash/utils.sh
source bash/set_datasets.sh

print_info_msg "Eroding datasets..."
for dataset in "${DATASETS[@]}"; do
    print_info_msg "Processing "$dataset
    python3 erode_dataset.py --dataset=$dataset --kernel=3 --iters=2 --viz=0
    check_success $?
done
