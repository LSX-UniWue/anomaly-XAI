
import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
from torchvision import transforms


# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


class MVTecData:
    def __init__(self, root, load_size, input_size):
        self.SHUFFLE = {'train': True,
                        'val': False,
                        'test': False,
                        'anom': False,
                        'ground_truth': False}

        self.root = root
        self.data_transforms = transforms.Compose([
            transforms.Resize((load_size, load_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((load_size, load_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(input_size)])

        self.train_data = MVTecDataset(root=self.root,
                                       data_transforms=self.data_transforms,
                                       gt_transforms=self.gt_transforms,
                                       phase='train')
        self.test_data = MVTecDataset(root=self.root,
                                      data_transforms=self.data_transforms,
                                      gt_transforms=self.gt_transforms,
                                      phase='test')
        anom_idx = [i for i in range(len(self.test_data)) if self.test_data.labels[i] != 0]
        self.anom_data = torch.utils.data.Subset(self.test_data, anom_idx)

    def get_data_loader(self, dataset_name, batch_size, shuffle=None):
        datasets = {'train': self.train_data,
                    'test': self.test_data,
                    'anom': self.anom_data}
        if shuffle is None:
            shuffle = self.SHUFFLE[dataset_name]
        return torch.utils.data.DataLoader(dataset=datasets[dataset_name],
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=0)

    def get_full_dataset(self, dataset_name):
        datasets = {'train': self.train_data,
                    'test': self.test_data,
                    'anom': self.anom_data}
        ds = next(iter(self.get_data_loader(dataset_name=dataset_name,
                                            batch_size=len(datasets[dataset_name]),
                                            shuffle=self.SHUFFLE[dataset_name])))
        return ds[0].numpy(), ds[1].numpy(), ds[2].numpy()  # img, ground_truth, label


class MVTecDataset(Dataset):
    def __init__(self, root, data_transforms, gt_transforms, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = data_transforms
        self.gt_transform = gt_transforms
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type
