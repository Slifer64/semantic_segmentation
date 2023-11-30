#! /bin/bash

object_data_path="data/object_datasets"

declare -A dataset_itrng=(
    ["$object_data_path/blue_cyl_dataset/"]="0:1"
    ["$object_data_path/bvase_dataset/"]="0:2"
    ["$object_data_path/lipton_box_dataset/"]="0:1"
    ["$object_data_path/oatbisc_box_dataset/"]="0:2"
    ["$object_data_path/wd40_dataset/"]="0:2"
    ["$object_data_path/brown_box_dataset/"]="0:2"
    ["$object_data_path/digestive_dataset/"]="0:2"
    ["$object_data_path/miwatch_box_dataset/"]="0:2"
    ["$object_data_path/uhu_dataset/"]="0:2"
)

DATASETS=()
for dataset in "${!dataset_itrng[@]}"; do
    DATASETS+=("$dataset")
done

all_datasets=""
items_range=""
for dataset in "${!dataset_itrng[@]}"; do
    all_datasets=$all_datasets" "$dataset
    items_range=$items_range" "${dataset_itrng[$dataset]}
done