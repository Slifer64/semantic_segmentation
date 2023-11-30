#! /bin/bash

MODEL='models/resnet18_seg.bin'
EVAL_SET='data/test_set/'
EVAL_SET='data/grasp_obj_train_set/'

# python3 eval_segmentation_model.py --model=$MODEL --dataset=$EVAL_SET --batch_size=6

python3 test_model_on_dataset.py --model=$MODEL --dataset=$EVAL_SET

# python3 test_model_in_RT.py --model=$MODEL --seg_cfg_path=$EVAL_SET
