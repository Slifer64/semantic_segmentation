#! /bin/bash

source bash/utils.sh

MODEL='models/resnet18_seg.bin'
TRAIN_SET='data/train_set/'
DEV_SET='data/dev_set/'
TEST_SET='data/test_set/'

print_info_msg "Training model "$MODEL
python3 train_segmentation_model.py --model=$MODEL --train_set=$TRAIN_SET --dev_set=$DEV_SET --test_set=$TEST_SET --epochs=200 --batch_size=16 --seed=9416
check_success $?
