import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


class FlowDataloader(Dataset):
    def __init__(self, file_dir, scale=1, shift=False, mode='disp', training_size=896, pin_mem=False):
        self.all_files = os.listdir(file_dir)
        self.all_inputs = [os.path.join(file_dir, file) for file in self.all_files if file.endswith(".npy")]
        self.all_sets = []
        self.blur5 = transforms.GaussianBlur(5, 5)
        self.shift = shift
        self.pin_mem = pin_mem
        self.training_size = training_size
        self.mode = mode
        self.scale = scale
        if pin_mem:
            for fname in self.all_inputs:
                array = np.load(fname)
                array_size = array.shape
                if min(array_size[1:]) <= training_size:
                    stpx = 0
                    stpy = 0
                else:
                    stpx = np.random.randint(0, array_size[1] - training_size)
                    stpy = np.random.randint(0, array_size[2] - training_size)
                array = array[:, stpx:stpx+training_size, stpy:stpy + training_size]
                imgs = torch.from_numpy(array[:2, :, :].astype('float32'))   # refimg, defimg
                imgs = self.blur5(imgs)
                # label = torch.from_numpy(array[2:, :, :].astype('float32'))    # flow_x, flow_y
                if shift:
                    label = torch.from_numpy(np.array([array[3, :, :], array[2, :, :]]).astype('float32'))      # 👉, 👇  flow_y, flow_x
                else:
                    label = torch.from_numpy(array[2:, :, :].astype('float32'))                                 # 👇, 👉  flow_x, flow_y
                if mode == 'strain':
                    disp_label = copy.deepcopy(label)
                    label = label.numpy()
                    kernel = np.array([[-0.25, 0, 0.25], [-0.5, 0, 0.5], [-0.25, 0, 0.25]])*0.5
                    kernel_y = kernel.T
                    strain_xx = cv2.filter2D(label[0], -1,  kernel)
                    strain_xy = cv2.filter2D(label[0], -1,  kernel_y)
                    strain_yx = cv2.filter2D(label[1], -1,  kernel)
                    strain_yy = cv2.filter2D(label[1], -1,  kernel_y)
                    label = torch.from_numpy(np.stack([strain_xx, strain_xy, strain_yx, strain_yy], axis=0))
                    label = [label, disp_label]
                temp_mask = array[0, :, :] > 0.1
                mask = torch.from_numpy(temp_mask.astype('float32')).unsqueeze(0)    # flow_x, flow_y
                min_gray = torch.min(imgs[1][300:500, 300:500])
                max_gray = torch.max(imgs[1][300:500, 300:500])
                ratio = 2.0 / (max_gray - min_gray)
                array_torch = (imgs - min_gray) * ratio - 1.0
                if scale == 2:
                    array_torch = F.interpolate(array_torch.unsqueeze(0), size=(array_size[1]//scale, array_size[2]//scale), mode='bilinear')
                    label = F.interpolate(label.unsqueeze(0), size=(array_size[1]//scale, array_size[2]//scale), mode='bilinear') / scale
                    mask = F.interpolate(mask.unsqueeze(0), size=(array_size[1]//scale, array_size[2]//scale), mode='bilinear')
                    self.all_sets.append([array_torch[0], label[0], mask[0]])
                else:
                    self.all_sets.append([array_torch, label, mask])
            # print(fname)

    def __len__(self):
        return int(len(self.all_inputs)//1)

    def __getitem__(self, item):
        if self.pin_mem:
            data_item = self.all_sets[item]
        else:
            fname = self.all_inputs[item]
            array = np.load(fname)
            array_size = array.shape
            if min(array_size[1:]) <= self.training_size:
                stpx = 0
                stpy = 0
            else:
                stpx = np.random.randint(0, array_size[1] - self.training_size)
                stpy = np.random.randint(0, array_size[2] - self.training_size)
            array = array[:, stpx:stpx + self.training_size, stpy:stpy + self.training_size]
            imgs = torch.from_numpy(array[:2, :, :].astype('float32'))  # refimg, defimg
            imgs = self.blur5(imgs)
            # label = torch.from_numpy(array[2:, :, :].astype('float32'))    # flow_x, flow_y
            if self.shift:
                label = torch.from_numpy(
                    np.array([array[3, :, :], array[2, :, :]]).astype('float32'))  # 👉, 👇  flow_y, flow_x
            else:
                label = torch.from_numpy(array[2:, :, :].astype('float32'))  # 👇, 👉  flow_x, flow_y
            if self.mode == 'strain':
                disp_label = copy.deepcopy(label)
                label = label.numpy()
                kernel = np.array([[-0.25, 0, 0.25], [-0.5, 0, 0.5], [-0.25, 0, 0.25]]) * 0.5
                kernel_y = kernel.T
                strain_xx = cv2.filter2D(label[0], -1, kernel)
                strain_xy = cv2.filter2D(label[0], -1, kernel_y)
                strain_yx = cv2.filter2D(label[1], -1, kernel)
                strain_yy = cv2.filter2D(label[1], -1, kernel_y)
                label = torch.from_numpy(np.stack([strain_xx, strain_xy, strain_yx, strain_yy], axis=0))
                label = [label, disp_label]
            temp_mask = array[0, :, :] > 0.1
            mask = torch.from_numpy(temp_mask.astype('float32')).unsqueeze(0)  # flow_x, flow_y
            min_gray = torch.min(imgs[1][300:500, 300:500])
            max_gray = torch.max(imgs[1][300:500, 300:500])
            ratio = 2.0 / (max_gray - min_gray)
            array_torch = (imgs - min_gray) * ratio - 1.0
            if self.scale == 2:
                array_torch = F.interpolate(array_torch.unsqueeze(0),
                                            size=(array_size[1] // self.scale, array_size[2] // self.scale), mode='bilinear')
                label = F.interpolate(label.unsqueeze(0), size=(array_size[1] // self.scale, array_size[2] // self.scale),
                                      mode='bilinear') / self.scale
                mask = F.interpolate(mask.unsqueeze(0), size=(array_size[1] // self.scale, array_size[2] // self.scale),
                                     mode='bilinear')
                data_item = [array_torch[0], label[0], mask[0]]
            else:
                data_item = [array_torch, label, mask]

        if self.mode == 'strain':
            return data_item[0], data_item[1], data_item[2], self.all_inputs[item]
        flow = torch.zeros_like(data_item[1], dtype=torch.float32)
        # 随机翻转
        rand = np.random.rand()
        if rand < 0.3:
            img = data_item[0].flip(1)
            mask = data_item[2].flip(1)
            if self.shift:
                flow[0, :, :] = data_item[1][0, :, :].flip(0)
                flow[1, :, :] = -data_item[1][1, :, :].flip(0)
            else:
                flow[0, :, :] = -data_item[1][0, :, :].flip(0)
                flow[1, :, :] = data_item[1][1, :, :].flip(0)
        elif rand < 0.6:        # 垂直于x👇方向翻转，左右翻转
            img = data_item[0].flip(2)
            mask = data_item[2].flip(2)
            if self.shift:
                flow[0, :, :] = -data_item[1][0, :, :].flip(1)      # 👉光流翻转、变号
                flow[1, :, :] = data_item[1][1, :, :].flip(1)       # 👇光流翻转、不变号
            else:
                flow[0, :, :] = data_item[1][0, :, :].flip(1)       # 👇光流翻转、不变号
                flow[1, :, :] = -data_item[1][1, :, :].flip(1)      # 👉光流翻转、变号
        else:
            img = data_item[0]
            flow = data_item[1]
            mask = data_item[2]
        return img, flow, mask, self.all_inputs[item]


def GenerateDataloader(file_dir, type="Train", batch_size=16, shuffle=True, scale=1, shift=False, mode='disp', pin_mem=False, training_size=768, p=True):
    return DataLoader(FlowDataloader(os.path.join(file_dir, type), scale=scale, shift=shift, mode=mode, pin_mem=pin_mem, training_size=training_size), batch_size=batch_size, shuffle=shuffle, num_workers=2)
