import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from my_pkg.models import ResnetSegmentation
from my_pkg.dataset import SegmentationDataset
from my_pkg.data_types import InputImage, SegmentationMask
import my_pkg.train_metrics as seg_metric

plt.ion()

def eval_segmentation_model(model_filename, dataset_path, batch_size, seed):

    torch.manual_seed(seed)

    model = ResnetSegmentation.load(model_filename)
    model.eval()
    model.train_history.plot()
    plt.pause(0.001)

    n_classes = model.n_classes

    dataset = SegmentationDataset.from_folder(dataset_path, data_transforms=None)
    # You can change the names of your image and mask, e.g. 'my_rgb_image.png', 'my_seg_mask.png'
    # dataset.in_cfg = {'rgb.png': InputImage}
    # dataset.out_cfg = {'mask.png': SegmentationMask}
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    plt.ion()

    min_IoU = 92

    for test_input, test_target in eval_loader:
        
        bad_result = False

        rgb_img = test_input[InputImage.data_name]
        seg_mask = test_target[SegmentationMask.data_name]
        batch_size = rgb_img.shape[0]
        with torch.no_grad():
            pred_out = model(rgb_img)
            pred_seg_mask = pred_out['seg_mask']
            pred_class = torch.argmax(pred_seg_mask, dim=1)


        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(batch_size, 3, width_ratios=[1, 2, 1], wspace=0.025, hspace=0.18, figure=fig)
        for i in range(batch_size):
            ax = [[plt.subplot(gs[i,j]) for j in range(3)] for i in range(batch_size)]
        for ax_row in ax:
            for ax_i in ax_row: ax_i.axis('off')

        y_true = torch.nn.functional.one_hot(seg_mask, n_classes)
        y_pred = torch.nn.functional.one_hot(pred_class, n_classes)

        for k in range(batch_size):
            seg_diff = (pred_class[k] != seg_mask[k]).numpy()
            # acc = seg_metric.accuracy(seg_mask[k], pred_class[k]) * 100.
            IoU = seg_metric.jaccard_index(y_true[k], y_pred[k]) * 100.

            if IoU < min_IoU: bad_result = True
            
            print_color = '\033[1;31m' if IoU < min_IoU else ''
            print(f'{print_color}{model.__class__.__name__} accuracy: {IoU:.2f} %\033[0m')
            
            img = rgb_img[k].numpy().transpose(1, 2, 0)
            ax[k][0].imshow(img)
            mask_separate = -1*torch.ones((seg_mask[k].shape[0], 10))
            masks = torch.concat((seg_mask[k], mask_separate, pred_class[k]), dim=1).numpy()
            ax[k][1].imshow(masks)
            ax[k][2].imshow(seg_diff, cmap='gray')
            title_opt = {'color': (1, 0, 0), 'fontweight': 'bold'} if IoU < min_IoU else {'color': (0, 0, 0)}
            ax[k][2].set_title(f"IoU: {IoU:.2f} %", **title_opt)
        ax[0][0].set_title('input and unveil trajectory')
        ax[0][1].set_title('groundtruth vs prediction')
        plt.pause(0.001)
        if bad_result or True:
            print("To continue press [enter]. To stop type 's' and press [enter]")
            if input().lower() == 's': break
        plt.close(fig)
        

# ==========================================
# =================== MAIN =================
# ==========================================

# Example usage:
# python3 eval_segmentation_model.py --model=models/resnet18_seg_model.bin --dataset=data/grapebunch_dataset/


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str,
                        help='The name of the segmentation model')
    parser.add_argument('--dataset', required=True, type=str,
                        help='The folder with the dataset')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Number of sampes to process/vizualize at each iteration.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for random operations.')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    args = parse_args()
    eval_segmentation_model(args['model'], args['dataset'], args['batch_size'], args['seed'])
