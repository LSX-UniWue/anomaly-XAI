
import os
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl

from data.MVTec.studentTeacherDataset import MVTecData
from anomaly_detection.mvtec_teacher_student import STPM, auto_select_weights_file
from xai.util import image_reference_points
from xai.automated_background_torch import optimize_local_input_gradient_descent


def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train', 'test'], default='test')
    parser.add_argument('--dataset_path', default=r'.\data\MVTec')
    parser.add_argument('--category', default='grid')
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.0001)
    parser.add_argument('--batch_size', default=256)  # 32
    parser.add_argument('--load_size', default=256)  # 256
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--project_path',
                        default=r'D:\IdeaProjects\code-2020-erp-xai\220906\test')
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--amap_mode', choices=['mul', 'sum'], default='mul')
    parser.add_argument('--val_freq', default=5)
    parser.add_argument('--weights_file_version', type=str, default=None)
    args = parser.parse_args()
    return args


def plot_img(img, segmentation_img=None, cmap='gray', alpha=0.35, save_path=None):
    fig, ax = plt.subplots()
    if img.shape[0] == 3:
        ax.imshow(np.transpose(img, [1, 2, 0]))
    elif img.shape[-1] == 3:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=cmap)
    if segmentation_img is not None:
        ax.imshow(segmentation_img, cmap='copper', alpha=alpha)
    ax.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=400)
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('GPU:', torch.cuda.is_available())
    args = get_args()

    reference_type = 'NN'  # one of ['zeros', 'NN', 'optimized', 'part-optimized']

    """Data loading"""
    mvtec = MVTecData(root=os.path.join(args.dataset_path, args.category),
                      load_size=args.load_size,
                      input_size=args.input_size)
    X_expl, ground_truth_data, y_expl = mvtec.get_full_dataset(dataset_name='anom')
    # plot_img(np.stack([item[0] for item in mvtec.train_data]).mean(axis=0))

    """Load the model"""
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=os.path.join(args.project_path, args.category),
                                            max_epochs=args.num_epochs,
                                            gpus=[0])
    # select weight file; select latest weight if args.weights_file_version == None
    weights_file_path = auto_select_weights_file(args.weights_file_version,
                                                 project_path=args.project_path,
                                                 category=args.category)
    if weights_file_path is not None:
        model = STPM(hparams=args, dataset=mvtec).load_from_checkpoint(weights_file_path, dataset=mvtec)
        model = model.to('cuda')
    else:
        raise ValueError('Weights file is not found!')

    data_point = X_expl[0]

    # TODO: Set example segment of perturbed values here
    segment = np.zeros(data_point.shape)
    segment[:, 60:220, 40:210] = 1

    plot_img(data_point, segmentation_img=segment.max(0))  # , save_path='./grid_x_with_mask.jpg')
    data_point = data_point.reshape([1, *data_point.shape])
    if reference_type in ['zeros', 'NN']:
        reference_points = image_reference_points(background=reference_type,
                                                  X_expl=data_point,
                                                  mvtec_data=mvtec,
                                                  predict_fn=model.forward,
                                                  device=device)
        plot_img(data_point[0], save_path='./grid_x.jpg')
        plot_img(reference_points[0], save_path=f'./grid_r_{reference_type}.jpg')
        perturbed_point = reference_points[0] * segment + data_point[0] * (1 - segment)
        plot_img(perturbed_point, save_path=f'./grid_x_dash_{reference_type}.jpg')
    elif reference_type == 'optimized':
        reference_points = image_reference_points(background=reference_type,
                                                  X_expl=data_point,
                                                  mvtec_data=mvtec,
                                                  predict_fn=model.forward,
                                                  device=device)
        plot_img(data_point[0])
        plot_img(reference_points[0], save_path=f'./grid_r_{reference_type}.jpg')
        difference = (data_point[0] - reference_points[0]) * 100
        plot_img(difference, save_path=f'./grid_x_x_dash_difference_{reference_type}.jpg')
    elif reference_type == 'part-optimized':
        # optimize data point with mask
        perturbed_point = optimize_local_input_gradient_descent(data_point=data_point,
                                                                mask=1 - segment,
                                                                predict_fn=model.forward,
                                                                max_iter=10000,
                                                                lr=0.001,
                                                                gamma=0.01,
                                                                flip_channels=False,
                                                                device=device,
                                                                verbose=1)
        plot_img(data_point[0])  # x
        plot_img(perturbed_point[0], save_path=f'./grid_x_dash_{reference_type}.jpg')  # x'
        plot_img((data_point[0] - perturbed_point[0]) * 100, save_path=f'./grid_x_x_dash_difference_{reference_type}.jpg')  # difference
    else:
        raise ValueError(f'unknown reference type: {reference_type}')
