import time

import numpy as np
from networks.gmflow.gmflow import GMFlow
from networks.SubpixelCorrNet import SubpixelCorrNet
from networks.DifferentialNet import DifferNet, DifferNet_FIT_Inter
from utils.interpolation import interpolator
import os
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import transforms
import torch.nn.functional as F


class StrainNet_LD():
    def __init__(self, params_dir='params/', device='cuda', calculate_strain=False, light_adjust=True,
                 fine_correlator='SubpixelCorrNet', strain_with_net=False, strain_radius=12, strain_skip=2,
                 raw_kernel_size=35, strain_decomp=True, num_iter=1):
        self.device = device
        self.raw_correlator = GMFlow(feature_channels=128,
                                     num_scales=1,
                                     upsample_factor=8,
                                     num_head=1,
                                     attention_type='swin',
                                     ffn_dim_expansion=4,
                                     num_transformer_layers=6,
                                     inchannel=3,
                                     ).eval()
        self.raw_correlator.load_state_dict(torch.load(os.path.join(params_dir, 'GMFlowNet_Best.pth')))
        self.raw_correlator.to(self.device).eval()
        if fine_correlator == 'SubpixelCorrNet':
            self.fine_correlator = SubpixelCorrNet()
            # self.fine_correlator.load_state_dict(torch.load(os.path.join(params_dir, 'SubpixelCorrNet_Best.pth')))
            self.fine_correlator.load_state_dict(torch.load(os.path.join(params_dir, 'SubpixelCorrNet_init.pth')))
            self.fine_correlator.to(self.device).eval()
        else:
            self.fine_correlator = None
        self.adjust_light = light_adjust
        if calculate_strain:
            self.strain_differlayer = DifferNet_FIT_Inter(device=device, radius=strain_radius, skip=strain_skip)
            if strain_with_net:
                self.strain_differlayer.load_state_dict(torch.load(os.path.join(params_dir, 'DifferNet_INT_Last.pth')))
            self.strain_differlayer.to(self.device).eval()
        self.interpolator = interpolator(device=self.device)
        self.raw_disp_blur = transforms.GaussianBlur(kernel_size=(raw_kernel_size, raw_kernel_size), sigma=raw_kernel_size//2+1)
        self.fine_disp_blur = transforms.GaussianBlur(kernel_size=(27, 27), sigma=14)
        self.strain_blur = transforms.GaussianBlur(kernel_size=(27, 27), sigma=15)
        self.strain_blur_2 = transforms.GaussianBlur(kernel_size=(25, 25), sigma=13)
        # self.fine_disp_blur = transforms.GaussianBlur(kernel_size=(11, 11), sigma=8)
        # self.strain_blur = transforms.GaussianBlur(kernel_size=(13, 13), sigma=7)
        # self.strain_blur_2 = transforms.GaussianBlur(kernel_size=(13, 13), sigma=13)
        self.img_blur = transforms.GaussianBlur(5, 5)
        self._avg_value = 120
        self.ref_config = None
        self.calculate_strain = calculate_strain
        self.strain_with_net = strain_with_net
        self.scale = 1
        self.strain_decomp = strain_decomp
        self.num_iter = num_iter


    def correlate_imgpair(self, img_ref, img_cur, mask=None, raw=False, scales=(1, 1)):      # 直接相关两张图像
        self.img_size = img_ref.shape
        # self.scale = max(self.img_size) // 1024 + 1

        # 图像对预处理
        imgs = np.stack([img_ref, img_cur], axis=0).astype('float32')
        ref_img, cur_img = self.__pre_process__(imgs)     # [1, 1, H, W]
        # 位移计算
        # cat_disp, raw_disp, fine_disp, imgs_inter = self.__correlation__(ref_img.to(self.device), cur_img.to(self.device), mask=mask)
        cat_disp, raw_disp, fine_disp, imgs_inter = self.__correlation_multi__(ref_img.to(self.device), cur_img.to(self.device), mask=mask)

        u = cat_disp[0].cpu().numpy()
        v = cat_disp[1].cpu().numpy()
        if self.calculate_strain:
            strain = self.__calculate_strain__(cat_disp, raw_disp, fine_disp, imgs_inter, scales)
            if raw:
                return u, v, raw_disp[0].cpu().numpy(), raw_disp[1].cpu().numpy(), strain
            return u, v, strain
        return u, v, None


    def update_refimg(self, img_ref, mask=None):       # 更新参考构型
        self.img_size = img_ref.shape
        # self.scale = max(self.img_size) // 1024 + 1
        # self.scale = max(self.img_size) // 1024 + 1
        img_ref = self.__pre_precess_single__(img_ref)
        self.ref_config = img_ref
        self._mask = mask


    def correlate_sequence(self, img_cur, scales=(1, 1)):    # 单个参考构型预测一系列图像
        if self.ref_config is None:
            return '没有设定参考构型'
        # 预处理变形构型
        img_cur = self.__pre_precess_single__(img_cur)
        cat_disp, raw_disp, fine_disp, imgs_inter = self.__correlation_multi__(self.ref_config.to(self.device), img_cur.to(self.device), mask=self._mask)
        u = cat_disp[0].cpu().numpy()
        v = cat_disp[1].cpu().numpy()
        if self.calculate_strain:
            strain = self.__calculate_strain__(cat_disp, raw_disp, fine_disp, imgs_inter, scales)
            return u, v, strain
        # 位移计算
        return u, v, None

    def __correlation__(self, ref_img, cur_img, mask, debug=False):
        # 输入：[1, 2, H, W], on self.device
        img_size = ref_img.shape[2:]
        raw_disp = self.raw_correlator(ref_img, cur_img)
        raw_disp = raw_disp['flow_preds'][-1][0, :, :, :].detach()
        raw_disp = self.raw_disp_blur(raw_disp)

        # 插值获得中间构型
        grid_x, grid_y = torch.meshgrid(torch.arange(img_size[0]), torch.arange(img_size[1]))
        grid_x, grid_y = grid_x.to(self.device), grid_y.to(self.device)
        u_pos = grid_x + raw_disp[1]
        v_pos = grid_y + raw_disp[0]

        img_inter = self.interpolator.interpolation(u_pos, v_pos, cur_img[0, 0, :, :], img_mode=False)

        # 精细像素位移计算
        if self.fine_correlator is not None:
            imgs_array = torch.stack([ref_img[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(self.device)
            fine_disp = self.fine_correlator(imgs_array)[0, :, :, :].detach()
        else:
            fine_disp = self.raw_correlator(ref_img, img_inter.unsqueeze(0).unsqueeze(0))['flow_preds'][-1][0, :, :, :].detach()

        # 插值获得精细大像素位移
        fine_disp = self.fine_disp_blur(fine_disp)
        [u_raw_inter, v_raw_inter] = self.interpolator.interpolation_list(
            u_pos=grid_x + fine_disp[1],
            v_pos=grid_y + fine_disp[0],
            gray_array_list=[raw_disp[0], raw_disp[1]],
            img_mode=False,
            kernel='bspline'
        )
        u = fine_disp[0] + u_raw_inter
        v = fine_disp[1] + v_raw_inter
        cat_disps = torch.stack([u, v], dim=0)
        if self.scale != 1:
            cat_disps = F.interpolate(cat_disps.unsqueeze(0), size=(img_size[0] * self.scale, img_size[1] * self.scale),
                                     mode='bilinear')
            cat_disps = cat_disps[0]

        if mask is not None:
            cat_disps = cat_disps * torch.from_numpy(mask.astype('float32')).to(self.device)
        if debug:
            u_raw = raw_disp[0].detach().cpu().numpy()
            v_raw = raw_disp[1].detach().cpu().numpy()
            if self.scale != 1:
                uv_raw_torch = F.interpolate(raw_disp.unsqueeze(0), size=(img_size[0]*self.scale, img_size[1]*self.scale), mode='bilinear')
                u_raw = uv_raw_torch[0, 0, :, :].numpy()
                v_raw = uv_raw_torch[0, 1, :, :].numpy()
            if mask is not None:
                u_raw = u_raw * mask
                v_raw = v_raw * mask
            return u, v, u_raw, v_raw

        return cat_disps, raw_disp, fine_disp, imgs_array

    def __correlation_multi__(self, ref_img, cur_img, mask, debug=False):
        # 输入：[1, 2, H, W], on self.device
        img_size = ref_img.shape[2:]
        raw_disp = self.raw_correlator(ref_img, cur_img)
        raw_disp = raw_disp['flow_preds'][-1][0, :, :, :].detach()
        raw_disp = self.raw_disp_blur(raw_disp)

        # 插值获得中间构型
        grid_x, grid_y = torch.meshgrid(torch.arange(img_size[0]), torch.arange(img_size[1]))
        grid_x, grid_y = grid_x.to(self.device), grid_y.to(self.device)

        tmp_u_raw = raw_disp[0]
        tmp_v_raw = raw_disp[1]

        for i in range(self.num_iter):

            u_pos = grid_x + tmp_v_raw
            v_pos = grid_y + tmp_u_raw

            tmp_cur_img = self.interpolator.interpolation(u_pos, v_pos, cur_img[0, 0, :, :], img_mode=False)

            # 精细像素位移计算
            tmp_fine_disp = self.fine_correlator(torch.stack([ref_img[0, 0, :, :], tmp_cur_img], dim=0).unsqueeze(0).to(self.device))[0, :, :, :].detach()

            # 插值获得精细大像素位移
            tmp_fine_disp = self.fine_disp_blur(tmp_fine_disp)
            [u_raw_inter, v_raw_inter] = self.interpolator.interpolation_list(
                u_pos=grid_x + tmp_fine_disp[1],
                v_pos=grid_y + tmp_fine_disp[0],
                gray_array_list=[tmp_u_raw, tmp_v_raw],
                img_mode=False,
                kernel='bspline'
            )
            if i < self.num_iter - 1:
                tmp_u_raw = tmp_fine_disp[0] + u_raw_inter
                tmp_v_raw = tmp_fine_disp[1] + v_raw_inter
            else:
                raw_disp = torch.stack([tmp_u_raw, tmp_v_raw], dim=0)

        u = tmp_fine_disp[0] + u_raw_inter
        v = tmp_fine_disp[1] + v_raw_inter

        cat_disps = torch.stack([u, v], dim=0)
        if self.scale != 1:
            cat_disps = F.interpolate(cat_disps.unsqueeze(0), size=(img_size[0] * self.scale, img_size[1] * self.scale),
                                     mode='bilinear')
            cat_disps = cat_disps[0]

        if mask is not None:
            cat_disps = cat_disps * torch.from_numpy(mask.astype('float32')).to(self.device)

        return cat_disps, raw_disp, tmp_fine_disp, torch.stack([ref_img[0, 0, :, :], tmp_cur_img], dim=0).unsqueeze(0).to(self.device)


    def __pre_precess_single__(self, img):
        if self.adjust_light:
            avg_matrix, _ = self.__Get_Avg_Matrix__(img)
            img = self._avg_value * img / avg_matrix
            img = (img - 255) * (img <= 255) + 255
        img = torch.from_numpy(img.astype('float32')).unsqueeze(0)
        img = self.img_blur(img)
        min_gray = torch.min(img[0][self.img_size[0]//3:-self.img_size[0]//3, self.img_size[1]//3:-self.img_size[1]//3])
        max_gray = torch.max(img[0][self.img_size[0]//3:-self.img_size[0]//3, self.img_size[1]//3:-self.img_size[1]//3])
        ratio = 2.0 / (max_gray - min_gray)
        array_torch = (img - min_gray) * ratio - 1.0
        if self.scale == 2:
            array_torch = F.interpolate(array_torch.unsqueeze(0), size=(self.img_size[0]//self.scale, self.img_size[1]//self.scale), mode='bilinear')
        else:
            array_torch = array_torch.unsqueeze(0)
        return array_torch  #[1, 1, H, W]


    def __pre_process__(self, imgs):        # 对图像对做预处理
        if not self.adjust_light:
            imgs = torch.from_numpy(imgs.astype('float32'))
        else:
            img_0 = imgs[0, :, :]
            img_1 = imgs[1, :, :]

            # plt.imshow(img_0)
            # plt.show()
            # plt.imshow(img_1)
            # plt.show()
            avg_matrix, _ = self.__Get_Avg_Matrix__(img_0)
            img_0 = self._avg_value * img_0 / avg_matrix
            img_0 = (img_0 - 255) * (img_0 <= 255) + 255
            avg_matrix, _ = self.__Get_Avg_Matrix__(img_1)
            img_1 = self._avg_value * img_1 / avg_matrix
            img_1 = (img_1 - 255) * (img_1 <= 255) + 255
            # plt.imshow(img_0)
            # plt.show()
            # plt.imshow(img_1)
            # plt.show()
            imgs = torch.from_numpy(np.stack([img_0, img_1], axis=0).astype('float32'))
        imgs = self.img_blur(imgs)
        min_grays = [torch.min(imgs[1][self.img_size[0]*(11-i-1)//11:self.img_size[0]*(11-i)//11, self.img_size[1]*i//11:self.img_size[1]*(i+1)//11]).item() for i in range(10)] +\
                    [torch.min(imgs[0][self.img_size[0]*i//11:self.img_size[0]*(i+1)//11, self.img_size[1]*i//11:self.img_size[1]*(i+1)//11]).item() for i in range(10)]
        max_grays = [torch.max(imgs[1][self.img_size[0]*(11-i-1)//11:self.img_size[0]*(11-i)//11, self.img_size[1]*i//11:self.img_size[1]*(i+1)//11]).item() for i in range(10)] +\
                    [torch.max(imgs[0][self.img_size[0]*i//11:self.img_size[0]*(i+1)//11, self.img_size[1]*i//11:self.img_size[1]*(i+1)//11]).item() for i in range(10)]
        min_grays.sort()
        min_gray = min_grays[10]
        max_grays.sort()
        max_gray = max_grays[10]
        ratio = 2.0 / (max_gray - min_gray)
        array_torch = (imgs - min_gray) * ratio - 1.0

        if self.scale == 2:
            array_torch = F.interpolate(array_torch.unsqueeze(0), size=(self.img_size[0]//self.scale, self.img_size[1]//self.scale), mode='bilinear')
        else:
            array_torch = array_torch.unsqueeze(0)
        return array_torch[:, 0, :, :].unsqueeze(1),  array_torch[:, 1, :, :].unsqueeze(1) #[1, 2, H, W]


    def __calculate_strain__(self, u_final, u_raw_bi, u_fine, inter_imgs, scales):
        # u_fine = self.fine_disp_blur(u_fine)
        u_fine = u_fine.unsqueeze(0)
        u_final = u_final.unsqueeze(0)
        u_raw_bi = u_raw_bi.unsqueeze(0)
        if self.strain_decomp:
            # u_fine = self.fine_disp_blur(u_fine)
            d_u_fine__d_x_0 = self.strain_differlayer(u_fine, with_net=False)
            d_u_raw__d_x_raw = self.strain_differlayer(u_raw_bi, with_net=False)
            strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
        else:
            strain = self.strain_differlayer(u_final, with_net=False)

        if self.strain_with_net:
            strain = strain + self.strain_differlayer(inter_imgs, with_net=True)
        strain = self.strain_blur(strain[0, :, :, :]).unsqueeze(0)
        strain = self.strain_blur_2(strain[0, :, :, :]).unsqueeze(0)
        strain = strain.detach().cpu().numpy()
        u_x, u_y, v_x, v_y = strain[0, 0], strain[0, 1], strain[0, 2], strain[0, 3]
        u_y = u_y * (scales[1] / scales[0])
        v_x = v_x * (scales[0] / scales[1])
        exx = u_x + 0.5 * (u_x**2 + v_x**2)
        exy = 0.5 * (u_y + v_x + u_x*u_y + v_x*v_y)
        eyy = v_y + 0.5 * (u_y**2 + v_y**2)

        return exx, exy, eyy


    def __Get_Avg_Matrix__(self, img, windsize=100):    # 灰度调整
        imsize = img.shape
        avg_matrix = []
        avg_img = img.mean()
        for i in range(imsize[0] // windsize):
            start_x = i * windsize
            t_list = []
            for j in range(imsize[1] // windsize):
                start_y = j * windsize
                t_block = img[start_x:start_x + windsize, start_y:start_y + windsize]
                avg = np.average(t_block)
                if avg < 0.5 * avg_img:
                    t_list.append(avg_img)
                else:
                    t_list.append(avg)
            avg_matrix.append(t_list)
        avg_matrix = cv2.resize(np.array(avg_matrix, dtype="uint8"), dsize=(imsize[1], imsize[0]),
                                interpolation=cv2.INTER_CUBIC)
        return avg_matrix, np.average(avg_matrix)


class StrainNet_L3D(StrainNet_LD):
    def __init__(self, params_dir='../params/', device='cuda', raw_kernel_size=55, num_iter=1, blockpos=None):
        super(StrainNet_L3D, self).__init__(params_dir=params_dir, device=device, raw_kernel_size=raw_kernel_size, num_iter=num_iter, calculate_strain=False)
        with open(os.path.join(params_dir, 'Calib.json'), "r") as f:
            params = json.load(f)

        self.Tx_r = params["RTX"]
        self.Tz_r = params["RTZ"]
        self.cx_r = params["RCX"]
        self.fx_r = params["RFX"]

        self.cx_l = params["LCX"]
        self.cy_l = params["LCY"]
        self.fx_l = params["LFX"]
        self.fy_l = params["LFY"]
        self.RRot = np.array(params["RRot"])
        self.shift_LY, self.shift_LX = np.meshgrid(range(0, params['LResolution'][1]), range(0, params['LResolution'][0]))
        if blockpos is not None:
            self.shift_LX = self.shift_LX[blockpos[0]:blockpos[1], blockpos[2]:blockpos[3]]
            self.shift_LY = self.shift_LY[blockpos[0]:blockpos[1], blockpos[2]:blockpos[3]]

    def calc_3DFlow(self, L0=None, L1=None, R0=None, R1=None, mask=None, disparity_0=None, flow_x=None, flow_y=None, fd_x=None, imsize=None, debug=False):
        if fd_x is None:
            L0 = cv2.resize(L0, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            L1 = cv2.resize(L1, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            R0 = cv2.resize(R0, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            R1= cv2.resize(R1, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            mask= cv2.resize(mask, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            rescale_factor = (imsize[0] / 1024, imsize[1] / 1024)
            disparity_0, _, _ = self.correlate_imgpair(L0, R0, mask)
            flow_x, flow_y, _ = self.correlate_imgpair(L0, L1, mask)
            fd_x, _, _ = self.correlate_imgpair(L0, R1, mask)
            disparity_0 = cv2.resize(disparity_0 * rescale_factor[1], dsize=imsize, interpolation=cv2.INTER_CUBIC)
            flow_x = cv2.resize(flow_x * rescale_factor[1], dsize=imsize, interpolation=cv2.INTER_CUBIC)
            flow_y = cv2.resize(flow_y * rescale_factor[0], dsize=imsize, interpolation=cv2.INTER_CUBIC)
            fd_x = cv2.resize(fd_x * rescale_factor[1], dsize=imsize, interpolation=cv2.INTER_CUBIC)

        x0, y0, z0 = self.recon_3d(disparity_0, flow_y=0)
        x1, y1, z1 = self.recon_3d(fd_x - flow_x, flow_y=flow_y)
        if debug:
            return [x1 - x0, y1 - y0, z1 - z0], [x0, y0, z0], [x1, y1, z1], [flow_x, flow_y], disparity_0
        return [x1 - x0, y1 - y0, z1 - z0], [x0, y0, z0], [x1, y1, z1]

    def calc_3DRecon(self, L0=None, R0=None, mask=None, disparity_0=None):
        imsize = L0.shape
        if disparity_0 is None:
            L0 = cv2.resize(L0, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            R0 = cv2.resize(R0, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            if mask is not None:
                mask= cv2.resize(mask, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            rescale_factor = (imsize[0] / 1024, imsize[1] / 1024)
            disparity_0, _, _ = self.correlate_imgpair(L0, R0, mask)
            disparity_0 = cv2.resize(disparity_0 * rescale_factor[1], dsize=imsize, interpolation=cv2.INTER_CUBIC)

        x0, y0, z0 = self.recon_3d(disparity_0, flow_y=0)
        return [x0, y0, z0]

    def recon_3d(self, disparity, flow_y=0):
        left_coord_y_1 = self.shift_LX + flow_y
        left_coord_x_1 = self.shift_LY
        right_coord_x_1 = left_coord_x_1 + disparity
        zw = (self.Tx_r - self.Tz_r * (right_coord_x_1 + 1.0 - self.cx_r) / self.fx_r) / \
               (((right_coord_x_1 + 1.0 - self.cx_r) / self.fx_r) * (
                       self.RRot[2, 0] * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l + self.RRot[2, 1] * (
                       left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l + self.RRot[2, 2]) -
                (self.RRot[0, 0] * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l + self.RRot[0, 1] * (
                        left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l +
                 self.RRot[0, 2]))
        xw = zw * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l
        yw = zw * (left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l
        return xw, yw, zw


class StrainNet_2D_Autoscale(StrainNet_LD):
    def __init__(self, params_dir='params/', device='cuda', calculate_strain=False, light_adjust=True,
                 fine_correlator='SubpixelCorrNet', strain_with_net=False, strain_radius=12, strain_skip=2,
                 raw_kernel_size=35, strain_decomp=True, num_iter=1):
        super(StrainNet_2D_Autoscale, self).__init__(params_dir=params_dir, device=device, calculate_strain=calculate_strain, light_adjust=light_adjust,
                 fine_correlator=fine_correlator, strain_with_net=strain_with_net, strain_radius=strain_radius, strain_skip=strain_skip,
                 raw_kernel_size=raw_kernel_size, strain_decomp=strain_decomp, num_iter=num_iter)


    def calc_2DFlow(self, I0=None, I1=None, mask=None, dsize=(1024, 1024)):
        imsize = I0.shape
        I0 = cv2.resize(I0, dsize=(dsize[1], dsize[0]), interpolation=cv2.INTER_CUBIC)
        I1 = cv2.resize(I1, dsize=(dsize[1], dsize[0]), interpolation=cv2.INTER_CUBIC)
        rescale_factor = (imsize[0] / dsize[0], imsize[1] / dsize[1])

        flow_x, flow_y, strain = self.correlate_imgpair(I0, I1, mask, scales=(rescale_factor[0], rescale_factor[1]))

        flow_x = cv2.resize(flow_x * rescale_factor[1], dsize=(imsize[1], imsize[0]), interpolation=cv2.INTER_CUBIC)
        flow_y = cv2.resize(flow_y * rescale_factor[0], dsize=(imsize[1], imsize[0]), interpolation=cv2.INTER_CUBIC)

        if self.calculate_strain:
            strain = [cv2.resize(f, dsize=(imsize[1], imsize[0]), interpolation=cv2.INTER_CUBIC) for f in strain]

        return flow_x, flow_y, strain


def correlate_image_pair():
    dataset_dir = r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\Eval'
    npy = np.load(os.path.join(dataset_dir, '17_img&disp.npy'))
    img_ref = npy[0, :, :]
    img_cur = npy[1, :, :]
    u_gt = npy[2, :, :]
    v_gt = npy[3, :, :]
    mask = img_ref > 0.1
    my_correlator = StrainNet_LD(device='cuda', calculate_strain=True)
    for i in range(5):
        now = time.perf_counter()
        u, v, strain = my_correlator.correlate_imgpair(img_ref, img_cur, mask=mask, debug=False)
        print('time consume: ', time.perf_counter() - now)
    plt.subplot(2, 2, 1)
    plt.imshow(u)
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(v)
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(npy[2, :, :])
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(npy[3, :, :])
    plt.colorbar()
    plt.show()
    plt.subplot(2, 2, 1)
    plt.imshow(strain[0], vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(strain[1], vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(strain[2], vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(strain[3], vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.show()
    print('error u: ', np.mean(np.abs(u - u_gt) * mask), 'error v: ', np.mean(np.abs(v - v_gt) * mask))
    # print('error ur: ', np.mean(np.abs(u_raw - u_gt) * mask), 'error vr: ', np.mean(np.abs(v_raw - v_gt) * mask))


def correlate_image_sequence():
    dataset_dir = r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\Eval'
    npy = np.load(os.path.join(dataset_dir, '17_img&disp.npy'))
    img_ref = npy[0, :, :]
    img_cur = npy[1, :, :]
    u_gt = npy[2, :, :]
    v_gt = npy[3, :, :]
    mask = img_ref > 0.1
    my_correlator = StrainNet_LD(device='cuda', calculate_strain=True, strain_with_net=True)
    my_correlator.update_refimg(img_ref, mask)
    for i in range(5):
        now = time.perf_counter()
        u, v, strain = my_correlator.correlate_sequence(img_cur)
        print('time consume: ', time.perf_counter() - now)
    plt.subplot(2, 2, 1)
    plt.imshow(u)
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(v)
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(npy[3, :, :])
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(npy[2, :, :])
    plt.colorbar()
    plt.show()
    plt.subplot(2, 2, 1)
    plt.imshow(strain[0], vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(strain[1], vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(strain[2], vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(strain[3], vmin=-.1, vmax=.1)
    plt.colorbar()
    plt.show()
    print('error u: ', np.mean(np.abs(u - u_gt) * mask), 'error v: ', np.mean(np.abs(v - v_gt) * mask))
    # print('error ur: ', np.mean(np.abs(u_raw - u_gt) * mask), 'error vr: ', np.mean(np.abs(v_raw - v_gt) * mask))


if __name__ == '__main__':
    correlate_image_pair()
    # correlate_image_sequence()




