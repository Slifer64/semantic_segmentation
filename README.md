# Contents
- [Description](#description)
- [Setup](#setup)
- [Collect/Create dataset](#collectcreate-dataset)
    - [Record object dataset](#record-object-dataset)
    - [Record backgrounds](#record-background-dataset)
    - [Merge and augment objects and backgrounds](#merge-and-augment-recorded-objects-with-backgrounds)
- [Train/eval segmentation model](#traineval-segmentation-model)
    - [Train](#train)
    - [Eval](#test)
    - [Test in RT](#test-in-real-time)


# Description
This package can be used to:
- **collect/create a custom dataset for semantic segmentation**: \
    Using a white (or other appropriate) background, you can extract masks for desired objects, using HSV color sliders to isolate your desired object and capture it at different poses. In this was you can collect sepate datasets for each desired object, where each dataset containts the RGB images and corresponding masks of each desired object.
- **edit, augment, merge this dataset**: \
    You can then edit, apply data augmentation (rotations, translations, color jitter etc.) and mix together your desired object datasets to create more composite scenes.
- **train, evaluate and test a segmentation model in RT**: \
    Using your final mixed dataset, you can train and test a semantic segmentation model. \
    This model is based on the paper [Fully Convolutional Networks for Semantic Segmentation]( 	
https://doi.org/10.48550/arXiv.1411.4038), and uses a Resnet18 as backbone.



# Setup

Download the repo:
```bash
git clone https://github.com/Slifer64/semantic_segmentation.git
cd semantic_segmentation/
```
Create a virtual environment:
```bash
python3 -m venv .env
```
Activate it and install the package dependencies:
```bash
source .env/bin/activate
pip install -e .
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

# Collect/Create dataset

## Record object dataset
Ideally, use a monochrome background (e.g. white or some other baclground that is distinc from your desired object). Using HSV color sliders, you can isolate your desired object.
```bash
# Option 1: use a real camera
python3 record_object_dataset.py --save_to=data/obj_dataset/ --label_id=obj:1 --i_start=1 --camera=rs
# Option 2: load images from a folder, e.g. "data/my_dataset"
python3 record_object_dataset.py --save_to=data/obj_dataset/ --label_id=obj:1 --i_start=1 --camera=bag:data/my_dataset/
```
Start recording samples, numbered from `i_start` (this way more samples can be added to an existing folder, without overwriting previous ones).
The format of `--label_id` should be `<object_name>:<object_id>`.
Press `"r" to record the current image and mask` (make sure that the cv window is selected).
Press `"q" to exit`.

### Erode dataset
Use this on a recorded dataset to reduce the borders of the segmentation masks (e.g. to ensure that no other pixels are accidentally included). This will modify the masks in the input `--dataset`, so it may be good to make a backup copy first. 
```bash
python3 erode_dataset.py --dataset=data/obj_dataset/ --kernel=3 --iters=1 --viz=0
```
Parameters:
- `--dataset`: path to the recorded dataset (**this dataset will be modified!**)
- `--kernel`: the size of the kernel used for eroding.
- `--iter`: number of times to apply the erosion.
- `--viz`: set to `1` if you want to vizualize the result for each image in the dataset.

## Record background dataset
Use to record a dataset of backgrounds:
```bash
python3 record_bg_dataset.py --save_to=data/bg_dataset/ --count=1
```

## Merge and augment recorded objects with backgrounds
Example:
```bash
python3 augmerge_dataset.py --datasets data/obj1_dataset/ data/obj2_dataset/ --items_range 1:3 4:10 --bg_dataset=data/bg_dataset/ --save_to=data/my_dataset/ --iters=4000 --seed=0 --viz=1
```
Merges the `obj1_dataset` and `obj2_dataset` using as background the `bg_dataset`, creating `--iters=4000` new images, where for each image `1-3` instances of `obj1` and `4-10` instances of `obj2` are randomly transformed and placed. More datasets can be specified after `--datasets` along with the corresponding `--items_range`. The samples picked from each dataset can be reused. Set `--iters` relatively high to ensure that all samples will be picked.

Based on your needs, you can customize the data augmentation options in [augmerge_dataset.py](./scripts/augmerge_dataset.py#L38):
```python
ALLOW_OVERLAP = False # whether to allow overlapping between object instances of the same class
MAX_N_OVERLAP = 3 # if ALLOW_OVERLAP==False, maximum attempts to add an object without overlapping, before skipping it

# object transforms: applied to each object instance separately
angle_range = [-80, 80]
tx_range = [-180, 180]
ty_range = [-250, 250]
scale_range = [0.8, 1.5]
p_hflip = 0.5

## Color jitter applies in the final mixed image, 
## so it may be too much to also apply separately color jitter to the bg.
# bg_color_jit = {'brightness': 0.2, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.05}

# global transforms: applied on the final mixed image (with different objects and background)
color_jit = {'brightness': [0.8, 1.6], 'contrast': [0.85, 1.2], 'saturation': [0.75, 1.4], 'hue': 0.04}
perspective_tf = {'p': 0.0, 'distortion': 0.5} # probablity of applying distortion and distortion scale
p_noise = 0.25 # probability of adding Gaussian noise
gaussian_noise = {'mean': 0.0, 'std': 0.1}
```

## View dataset
```bash
python3 view_dataset.py --dataset=data/my_dataset/
```

# Train/Eval segmentation model

## Train
Using the procedure descibed [above](#collectcreate-dataset) you can create a `train_dataset`, `dev_dataset` and optionally a `test_dataset` as well, and train a segmentation model:
```bash
python3 train_segmentation_model.py --model=models/my_model.bin --train_set=data/train_dataset/ --dev_set=data/dev_dataset/ --test_set=data/test_dataset/ --epochs=200 --batch_size=16 --seed=0
```
The training returns the model with the minimum dev-set loss. You 

## Eval
```bash
python3 eval_segmentation_model.py --model=models/my_model.bin --dataset=data/my_eval_dataset/ --batch_size=4
```

## Test in real-time
Test with:
- real camera
    ```bash
    python3 test_model_in_RT.py --model=models/my_model.bin --seg_cfg_path=data/train_dataset/
    ```
- custom dataset:
    ```bash
    python3 test_model_in_RT.py --model=models/my_model.bin --seg_cfg_path=data/train_dataset/ --camera=bag:data/test_dataset/
    ```

Notice that you have to specify the path to the config file that contains the object names and their ids. E.g. in the above examples we use the autogenerated (from the [data collection/creation step](#collectcreate-dataset)) that is stored in `data/train_dataset/`.

---