import os
import cv2
import glob
import shutil
import contextlib

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import resnet18
import pytorch_lightning as pl


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def prep_dirs(root):
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    return sample_path


def convert_inplace_relus(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.ReLU())
        else:
            convert_inplace_relus(child)


class STPM(pl.LightningModule):
    def __init__(self, hparams, dataset):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)
        self.args = hparams
        self.dataset = dataset

        self.init_features()

        def hook_t(module, input, output):
            self.features_t.append(output)

        def hook_s(module, input, output):
            self.features_s.append(output)

        self.model_t = resnet18(pretrained=True).eval()
        convert_inplace_relus(self.model_t)
        for param in self.model_t.parameters():
            param.requires_grad = False


        self.model_t.layer1[-1].register_forward_hook(hook_t)
        self.model_t.layer2[-1].register_forward_hook(hook_t)
        self.model_t.layer3[-1].register_forward_hook(hook_t)

        self.model_s = resnet18(pretrained=False)  # default: False
        convert_inplace_relus(self.model_s)
        # self.model_s.apply(init_weights)
        self.model_s.layer1[-1].register_forward_hook(hook_s)
        self.model_s.layer2[-1].register_forward_hook(hook_s)
        self.model_s.layer3[-1].register_forward_hook(hook_s)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

        self.inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

    def init_features(self):
        self.features_t = []
        self.features_s = []

    def forward_features(self, x):
        self.init_features()
        x_t = self.model_t(x)
        x_s = self.model_s(x)
        return self.features_t, self.features_s

    def forward(self, x, keep_grad=True, add_output_dim=True, flip_channels=False, device='cuda',
                output_cpu=False, output_to_numpy=False):
        """Joint method for scoring one batch with gradients for gradient xais"""
        if flip_channels:  # SHAP maskers only work with channels last, so we convert to channels first here
            if len(x.shape) == 4:
                x = x.permute(0, 3, 1, 2) if torch.is_tensor(x) else np.transpose(x, [0, 3, 1, 2])
            elif len(x.shape) == 3:
                x = x.permute(2, 0, 1) if torch.is_tensor(x) else np.transpose(x, [2, 0, 1])

        self.eval()
        with (torch.no_grad() if not keep_grad else contextlib.nullcontext()):
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            x = x.to(device)
            features_t, features_s = self.forward_features(x)
            # Get anomaly map
            # - features is list of 3 tensors, first dimension is batch_size
            anomaly_map, a_map_list = self.cal_anomaly_map(fs_list=features_s,
                                                           ft_list=features_t,
                                                           out_size=self.args['input_size'],
                                                           output_to_numpy=False,
                                                           batch_mode=True)

            # self.pred_list_px_lvl.extend(anomaly_map.ravel())  # segmentation map
            scores = torch.linalg.norm(anomaly_map, dim=(1, 2))

            if add_output_dim:  # need additional dimension for captum
                scores = scores.unsqueeze(1)
            if output_to_numpy:
                scores = scores.cpu().detach().numpy()
            elif output_cpu:
                scores = scores.cpu()
            return scores

    def cal_loss(self, fs_list, ft_list):
        tot_loss = 0
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            _, _, h, w = fs.shape
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            f_loss = (0.5 / (w * h)) * self.criterion(fs_norm, ft_norm)
            tot_loss += f_loss

        return tot_loss

    def cal_anomaly_map(self, fs_list, ft_list, out_size=224, output_to_numpy=True, batch_mode=False):
        if self.args['amap_mode'] == 'mul':
            if batch_mode:
                anomaly_map = np.ones([fs_list[0].shape[0], out_size, out_size]) if output_to_numpy else torch.ones([fs_list[0].shape[0], out_size, out_size]).to(fs_list[0].device)
            else:
                anomaly_map = np.ones([out_size, out_size]) if output_to_numpy else torch.ones([out_size, out_size]).to(fs_list[0].device)
        else:
            if batch_mode:
                anomaly_map = np.zeros([fs_list[0].shape[0], out_size, out_size]) if output_to_numpy else torch.zeros([fs_list[0].shape[0], out_size, out_size]).to(fs_list[0].device)
            else:
                anomaly_map = np.zeros([out_size, out_size]) if output_to_numpy else torch.zeros([out_size, out_size]).to(fs_list[0].device)
        a_map_list = []
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            fs_norm = F.normalize(fs, p=2)  # we norm over the channels; correct?
            ft_norm = F.normalize(ft, p=2)
            a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear')
            # here we are deleting the images in non-batch-mode (original test code with batch_size = 1)
            if batch_mode:
                a_map = a_map[:, 0, :, :]
            else:
                a_map = a_map[0, 0, :, :]
            if output_to_numpy:
                a_map = a_map.to('cpu').detach().numpy()
            a_map_list.append(a_map)
            if self.args['amap_mode'] == 'mul':
                anomaly_map *= a_map
            else:
                anomaly_map += a_map
        return anomaly_map, a_map_list

    def save_anomaly_map(self, anomaly_map, a_maps, input_img, gt_img, file_name, x_type):
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)
        # 64x64 map
        am64 = min_max_norm(a_maps[0])
        am64 = cvt2heatmap(am64 * 255)
        # 32x32 map
        am32 = min_max_norm(a_maps[1])
        am32 = cvt2heatmap(am32 * 255)
        # 16x16 map
        am16 = min_max_norm(a_maps[2])
        am16 = cvt2heatmap(am16 * 255)
        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm * 255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        # file_name = id_generator() # random id
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am64.jpg'), am64)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am32.jpg'), am32)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_am16.jpg'), am16)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model_s.parameters(), lr=self.args['lr'], momentum=self.args['momentum'],
                               weight_decay=self.args['weight_decay'])

    def train_dataloader(self):
        return self.dataset.get_data_loader(dataset_name='train', batch_size=self.args['batch_size'])

    def test_dataloader(self):
        return self.dataset.get_data_loader(dataset_name='test', batch_size=1)

    def on_train_start(self):
        self.model_t.eval()  # to stop running_var move (maybe not critical)
        self.sample_path = prep_dirs(self.logger.log_dir)

    def on_test_start(self):
        self.init_results_list()
        self.sample_path = prep_dirs(self.logger.log_dir)

    def training_step(self, batch, batch_idx):
        x, _, _, file_name, _ = batch
        features_t, features_s = self.forward_features(x)
        loss = self.cal_loss(features_s, features_t)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, gt, label, file_name, x_type = batch
        features_t, features_s = self.forward_features(x)

        # Get anomaly map
        anomaly_map, a_map_list = self.cal_anomaly_map(features_s, features_t, out_size=self.args['input_size'])

        gt_np = gt.cpu().numpy().astype(int)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(np.linalg.norm(anomaly_map))
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map, a_map_list, input_x, gt_np[0][0] * 255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)


def auto_select_weights_file(weights_file_version, project_path, category):
    print()
    version_list = glob.glob(os.path.join(project_path, category) + '/lightning_logs/version_*')
    version_list.sort(reverse=True, key=lambda x: os.path.getmtime(x))
    if weights_file_version != None:
        version_list = [os.path.join(project_path, category) + '/lightning_logs/' + weights_file_version] + version_list
    for i in range(len(version_list)):
        weights_file_path = glob.glob(os.path.join(version_list[i], 'checkpoints') + '/*')
        if len(weights_file_path) == 0:
            if weights_file_version != None and i == 0:
                print(f'Checkpoint of {weights_file_version} not found')
            continue
        else:
            weights_file_path = weights_file_path[0]
            if weights_file_path.split('.')[-1] != 'ckpt':
                continue
        print('Checkpoint found : ', weights_file_path)
        print()
        return weights_file_path
    print('Checkpoint not found')
    print()
    return None
