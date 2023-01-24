import os
import argparse
import functools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pytorch_lightning as pl
from torchinfo import summary

from xai.util import image_reference_points
from anomaly_detection.mvtec_teacher_student import STPM, auto_select_weights_file
from data.MVTec.studentTeacherDataset import MVTecData


"""
Complement to mvtec_shap.py that implements SHAP running with nearest neighbor references
"""


def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train', 'test'], default='test')
    parser.add_argument('--dataset_path', default=r'.\data\MVTec' if os.name == 'nt' else r'./data/MVTec')
    parser.add_argument('--category', default='grid')
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.0001)
    parser.add_argument('--batch_size', default=256)  # 32
    parser.add_argument('--load_size', default=256)  # 256
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--project_path',
                        default=r'.\220906\test' if os.name == 'nt' else r'./220906/test')
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--amap_mode', choices=['mul', 'sum'], default='mul')
    parser.add_argument('--val_freq', default=5)
    parser.add_argument('--weights_file_version', type=str, default=None)
    parser.add_argument('--start_point', default=0)  # SHAP start point
    parser.add_argument('--end_point', default=-1)  # SHAP end point, defaults to all anomalies
    args = parser.parse_args()
    return args


def plot_img(img, segmentation_img=None, cmap='gray', alpha=0.35):
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
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('GPU:', torch.cuda.is_available())
    args = get_args()

    xai_type = 'shap'
    background = 'NN'

    expl_path = f'outputs/explanation/{xai_type}_st/{args.category}.npy'
    out_path = 'outputs/explanation/mvtec_summary.csv'

    """Data loading"""
    mvtec = MVTecData(root=os.path.join(args.dataset_path, args.category),
                      load_size=args.load_size,
                      input_size=args.input_size)
    X_expl, ground_truth_data, y_expl = mvtec.get_full_dataset(dataset_name='anom')
    # plot_img(np.stack([item[0] for item in mvtec.train_data]).mean(axis=0))

    """Load / train the model"""
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=os.path.join(args.project_path, args.category),
                                            max_epochs=args.num_epochs,
                                            gpus=[0])
    if args.phase == 'train':
        model = STPM(hparams=args, dataset=mvtec)
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        # select weight file; select latest weight if args.weights_file_version == None
        weights_file_path = auto_select_weights_file(args.weights_file_version,
                                                     project_path=args.project_path,
                                                     category=args.category)
        if weights_file_path is not None:
            model = STPM(hparams=args, dataset=mvtec).load_from_checkpoint(weights_file_path, dataset=mvtec)
            # trainer.test(model)
        else:
            raise ValueError('Weights file is not found!')
    else:
        raise ValueError("args.phase not defined, should be one of ['train', 'test']")
    summary(model=model, input_size=(args.batch_size, *X_expl[0].shape))

    """Calculate explanations if not already saved"""
    if not os.path.exists(expl_path):
        if not os.path.exists(os.path.dirname(expl_path)):
            os.mkdir(os.path.dirname(expl_path))

        if xai_type == 'shap':
            import shap
            from shap.maskers._image import Image

            # here we explain two images using 10000 evaluations of the underlying model to estimate the SHAP values
            # SHAP maskers only work with channels last, so we permute from channels first and switch back at forward
            # shape (40, 3, 256, 256)
            x = X_expl
            x = np.transpose(x, [0, 2, 3, 1])
            model = model.to('cuda')
            if background == 'NN':
                NNs = image_reference_points(background=background,
                                             X_expl=X_expl,
                                             mvtec_data=mvtec)
            else:
                raise ValueError(f'only for shap + NN, use the other file!')
            detection_fn = functools.partial(model.forward,
                                             keep_grad=False,
                                             add_output_dim=False,
                                             flip_channels=True,
                                             output_to_numpy=True)

            # one data point at a time for intermediate results
            start_point, end_point = int(args.start_point), int(args.end_point)
            if end_point == -1:
                end_point = x.shape[0]
            for i in range(start_point, end_point):

                # TODO: this is the only change from mvtec_xai.py:
                #  - masker needs specific values for each input to explain
                #  -> need to re-instantiate masker -> need to re-instantiate explainer
                masker = Image(NNs[i], x[0].shape)
                explainer = shap.Explainer(detection_fn, masker)

                shard_expl_path = f'outputs/explanation/{xai_type}_st/{args.category}_{str(i)}.npy'
                shap_values = explainer(x[i].reshape([1, *x[i].shape]),
                                        max_evals=10000,  # 10k seems to give decent tradeoff for resolution / runtime
                                        batch_size=args.batch_size)
                # shap.image_plot(shap_values)
                expl = shap_values.values
                expl = np.transpose(expl, [0, 3, 1, 2])
                np.save(file=shard_expl_path, arr=expl)

        else:
            raise ValueError(f'only for shap + NN, please use mvtec_xai.py for other approaches!')

        np.save(file=expl_path, arr=expl)

    """Evaluate explanations"""
    only_positive_scores = False

    expl = np.load(expl_path)
    ground_truth = ground_truth_data[:, 0]
    if expl.shape[1] == 3:  # add up color channels
        expl = expl.sum(axis=1)  # Additive explanations use sum (SHAP github issue)
    if ground_truth_data[0].shape[1] == 3:  # add up color channels
        ground_truth = ground_truth.max(axis=1)
    if only_positive_scores:
        rectify_fn = np.vectorize(lambda x: max(0, x))
        expl = rectify_fn(expl)
    plot_img(X_expl[0])
    plot_img(expl[0])  # test plot
    plot_img(expl[0], segmentation_img=ground_truth[0])  # test plot

    expl = expl.reshape([expl.shape[0], -1])  # flatten images

    ground_truth = ground_truth.reshape([ground_truth.shape[0], -1]).astype(int)  # flatten ground truth

    """
    "explanation accuracy" after kauffmann2020
    "(measured as the cosine similarity between the ground-truth and the pixel-wise explanation). 
    Pixel-wise explanations are passed through a rectification function 
    so that the cosine similarity is always greater or equal to zero."
    expl_acc = mean( cos_sim( ground_truth, max(0, explanation) ) )
    """
    cos_sims = []
    auc_rocs = []
    for i in range(ground_truth.shape[0]):
        auc_roc = roc_auc_score(y_true=ground_truth[i], y_score=expl[i])
        auc_rocs.append(auc_roc)
        cos_sim = cosine_similarity(ground_truth[i].reshape(1, -1), expl[i].reshape(1, -1))
        cos_sims.append(cos_sim[0, 0])

    out_dict = {'xai': xai_type,
                'variant': background,
                f'ROC': np.mean(auc_rocs),
                f'ROC-std': np.std(auc_rocs),
                f'Cos': np.mean(cos_sims),
                f'Cos-std': np.std(cos_sims)}
    [print(key + ':', val) for key, val in out_dict.items()]

    # save outputs to combined result csv file
    if out_path:
        if os.path.exists(out_path):
            out_df = pd.read_csv(out_path, header=0)
        else:
            out_df = pd.DataFrame()
        out_df = out_df.append(out_dict, ignore_index=True)
        out_df.to_csv(out_path, index=False)
