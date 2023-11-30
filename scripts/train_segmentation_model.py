import numpy as np
import torch
import os
from my_pkg.models import ResnetSegmentation
from my_pkg.train_utils import train_segmentation_model
from my_pkg.dataset import SegmentationDataset

from utils import SegConfig
from my_pkg.util.timer import Timer

def train_model(save_model_as, train_set, dev_set, test_set, batch_size, epochs, seed):

    torch.manual_seed(seed)

    cfg = SegConfig.load(train_set)
    n_classes = cfg.n_classes + 1 # for the background

    model = ResnetSegmentation(n_classes=n_classes, seg_type=4, resnet_type="resnet18")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=40)

    # dataset = SegmentationDataset.from_folder(dataset_path, data_transforms=None)
    # n_data = len(dataset)
    # i_train = int(train_split * n_data + 0.5)
    # ind = torch.randperm(n_data).tolist()  # if shuffle else list(range(n_data))
    # train_loader = torch.utils.data.DataLoader(dataset.subset(ind[:i_train]), batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = torch.utils.data.DataLoader(dataset.subset(ind[i_train:]), batch_size=batch_size, shuffle=True, num_workers=4)
    
    train_loader = torch.utils.data.DataLoader(SegmentationDataset.from_folder(train_set), batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(SegmentationDataset.from_folder(dev_set), batch_size=batch_size, shuffle=True, num_workers=8) if dev_set else None
    test_loader = torch.utils.data.DataLoader(SegmentationDataset.from_folder(test_set), batch_size=batch_size, shuffle=True, num_workers=8) if test_set else None

    class_weights = torch.ones(n_classes, device='cuda')
    # class_weights[0] = 0.5 # place less weight to the background
    def seg_loss_fun(y_hat, y):
        return torch.nn.CrossEntropyLoss(weight=class_weights)(y_hat, y)

    timer = Timer()
    timer.tic()
    model: ResnetSegmentation = train_segmentation_model(
        model=model,
        optimizer=optimizer,
        loss_fun=seg_loss_fun,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        metrics=['loss', 'accuracy', 'IoU'],
        return_choice='best_val'
    )
    print('\33[1;36mElapsed time %.1f sec\33[0m' % timer.toc())

    fig, axes = model.train_history.plot()

    # Create the intermediate directories if they don't exist
    directory = os.path.dirname(save_model_as)
    if not os.path.exists(directory):
        os.makedirs(directory)

    model.save(save_model_as)
    input('Press [enter] to finish...')


# ==========================================
# =================== MAIN =================
# ==========================================

# Example usage:
# python3 train_segmentation_model.py --model=models/resnet18_seg_model.bin --dataset=data/grapebunch_dataset/ --train_split=0.7 --epochs=200 --batch_size=16 --seed=0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', required=True, type=str,
                        help='The model save filename.')
    parser.add_argument('--train_set', required=True, type=str,
                        help='The folder with the train dataset')
    parser.add_argument('--dev_set', type=str, default='',
                        help='The folder with the dev dataset (optional)')
    parser.add_argument('--test_set', type=str, default='',
                        help='The folder with the test dataset (optional)')
    parser.add_argument('--batch_size', required=True, type=int,
                        help='Training batch size.')
    parser.add_argument('--epochs', required=True, type=int,
                        help='Training epochs')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for all random operations')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    args = parse_args()
    train_model(save_model_as=args['model'], train_set=args['train_set'], dev_set=args['dev_set'], test_set=args['test_set'], 
                batch_size=args['batch_size'], epochs=args['epochs'], seed=args['seed'])
