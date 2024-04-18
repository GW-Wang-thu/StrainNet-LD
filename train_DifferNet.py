import copy

from networks.SubpixelCorrNet import SubpixelCorrNet
from networks.gmflow.gmflow import GMFlow
from networks.DifferentialNet import DifferNet, DifferNet_FIT, DifferNet_FIT_Inter
from utils.dataloader import GenerateDataloader
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.interpolation import interpolator
import torchvision.transforms as tf


def train_DifferNet_FIT_eval(device='cuda', savedir='params', num_epochs=100, dataset_dir=r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\\', img_size=(768, 768)):

    train_dataloader = GenerateDataloader(dataset_dir, type='Train', batch_size=1, shift=True, mode='strain', training_size=img_size[0])
    valid_dataloader = GenerateDataloader(dataset_dir, type='Valid', batch_size=1, shift=True, mode='strain', training_size=img_size[0])

    model_GM = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   inchannel=3,
                   ).to(device)
    model_GM.load_state_dict(torch.load(os.path.join(savedir, 'GMFlowNet_Best.pth')))
    model_GM.eval()
    model_SP = SubpixelCorrNet().to(device)
    model_SP.load_state_dict(torch.load(os.path.join(savedir, 'SubpixelCorrNet_Best.pth')))
    model_SP.eval()
    my_interpolator = interpolator(device=device)

    model = DifferNet_FIT(radius=12, skip=4).to(device).eval()
    model.load_state_dict(torch.load(savedir + '//DifferNet_FIT_Last.pth'))

    kernel_size = 35
    my_raw_disp_blur = tf.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=kernel_size//2+1)
    my_fine_disp_blur = tf.GaussianBlur(kernel_size=(3, 3), sigma=2)
    grid_x, grid_y = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
    grid_x, grid_y = grid_x.to(device), grid_y.to(device)

    steps = 0
    num_epochs -= steps
    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    # 训练模型
    for n in range(num_epochs):
        epoch = n + steps
        mean_train_loss = 0
        mean_train_loss_epe = 0
        mean_train_loss_wo_net = 0
        mean_train_loss_wo_net_epe = 0
        disp_mean_loss_rec = 0
        disp_mean_epe = 0
        for i, (images, labels, mask, fname) in enumerate(tqdm(train_dataloader, desc="Training", unit="iteration")):
            # if n == 0:
            #     continue
            labels, disp_label = labels
            labels = labels.to(device)
            disp_label = disp_label.to(device)
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            # 前向传播
            pred_flow = model_GM(ref_imgs, def_imgs,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1])
            mask = mask.to(device)
            mask[:, :, -20:, :] = False
            mask[:, :, :20, :] = False
            mask[:, :, :, -20:] = False
            mask[:, :, :, :20] = False

            raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
            raw_disp = my_raw_disp_blur(raw_disp)

            # 插值获得中间构型
            u_pos = grid_x + raw_disp[1]
            v_pos = grid_y + raw_disp[0]

            img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

            # 精细像素位移计算
            imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
            fine_disp = model_SP(imgs_array)
            # 插值获得精细大像素位移
            [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                u_pos=grid_x + fine_disp[0][1],
                v_pos=grid_y + fine_disp[0][0],
                gray_array_list=[raw_disp[0], raw_disp[1]],
                img_mode=False,
                kernel='bspline'
            )
            fine_disp = my_fine_disp_blur(fine_disp[0, :, :, :].detach())
            d_u_fine__d_x_0 = model(fine_disp.unsqueeze(0), with_net=False)
            d_u_raw__d_x_raw = model(raw_disp.unsqueeze(0), with_net=False)
            strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
            u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
            mask_value = torch.abs(u_raw_inter.detach() - disp_label) < 5.0
            mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
            pred_disp = fine_disp + u_raw_inter
            strain_res_pre = model(pred_disp, with_net=True)
            strain_pre = strain + strain_res_pre
            vmin = torch.min(labels[:, :, 2:-2, 2:-2]) - 1.0
            vmax = torch.max(labels[:, :, 2:-2, 2:-2]) + 1.0
            mask_value = (labels < vmax) * (labels > vmin)
            loss = torch.sum(torch.abs(strain_pre - labels) * mask_value * mask) / torch.sum(mask * mask_value) #+ torch.mean(torch.abs(disp_srefine - disp_label) * mask) * 1e-2
            if i == 0:
                plt.imshow((strain_pre * mask_value * mask)[0, 0, :, :].detach().cpu().numpy())
                plt.colorbar()
                plt.show()
                plt.imshow((labels * mask_value * mask)[0, 0, :, :].detach().cpu().numpy())
                plt.colorbar()
                plt.show()
                plt.imshow(ref_imgs[0, 0, :, :].detach().cpu().numpy())
                plt.show()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += torch.sum(torch.abs(strain_pre - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()
            mean_train_loss_wo_net += torch.sum(torch.abs(strain - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()

            mean_train_loss_epe += torch.sum(torch.sum((strain_pre.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_train_loss_wo_net_epe += torch.sum(torch.sum((strain.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

            disp_mean_loss_rec += torch.sum(torch.abs(pred_disp.detach() - disp_label) * mask).item() / torch.sum(mask).item()
            disp_mean_epe += torch.sum(torch.sum((pred_disp.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

        mean_train_loss = mean_train_loss / i
        mean_train_loss_wo_net = mean_train_loss_wo_net / i
        mean_train_loss_epe = mean_train_loss_epe / i
        mean_train_loss_wo_net_epe = mean_train_loss_wo_net_epe / i
        disp_mean_epe = disp_mean_epe / i
        disp_mean_loss_rec = disp_mean_loss_rec / i / 2

        # evaluator.show_seg_img()
        mean_loss = 0
        mean_loss_epe = 0
        absolute_strain = 0
        mean_loss_wo_net = 0
        mean_loss_wo_net_epe = 0
        disp_mean_loss_rec_valid = 0
        disp_mean_epe_valid = 0
        # disp_mean_epe_srefine_valid = 0
        for j, (images, labels, mask, fname) in enumerate(tqdm(valid_dataloader, desc="Validation", unit="iteration")):
            labels, disp_label = labels
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            # 前向传播
            pred_flow = model_GM(ref_imgs, def_imgs,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1])
            mask = mask.to(device)
            labels = labels.to(device)
            disp_label = disp_label.to(device)
            mask[:, :, -20:, :] = False
            mask[:, :, :20, :] = False
            mask[:, :, :, -20:] = False
            mask[:, :, :, :20] = False

            raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
            raw_disp = my_raw_disp_blur(raw_disp)

            # 插值获得中间构型
            u_pos = grid_x + raw_disp[1]
            v_pos = grid_y + raw_disp[0]
            img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

            # 精细像素位移计算
            imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
            fine_disp = model_SP(imgs_array)
            # 插值获得精细大像素位移
            [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                u_pos=grid_x + fine_disp[0][1],
                v_pos=grid_y + fine_disp[0][0],
                gray_array_list=[raw_disp[0], raw_disp[1]],
                img_mode=False,
                kernel='bspline'
            )

            fine_disp = my_fine_disp_blur(fine_disp[0, :, :, :].detach())
            d_u_fine__d_x_0 = model(fine_disp.unsqueeze(0), with_net=False)
            d_u_raw__d_x_raw = model(raw_disp.unsqueeze(0), with_net=False)
            strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
            u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
            mask_value = torch.abs(u_raw_inter.detach() - disp_label) < 5.0
            mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
            pred_disp = fine_disp + u_raw_inter
            strain_res_pre = model(pred_disp, with_net=True)
            strain_pre = strain + strain_res_pre

            vmin = torch.min(labels[:, :, 2:-2, 2:-2]) - 1.0
            vmax = torch.max(labels[:, :, 2:-2, 2:-2]) + 1.0
            mask_value = (labels < vmax) * (labels > vmin)
            loss = torch.sum(torch.abs(strain_pre - labels) * mask_value * mask) / torch.sum(mask * mask_value)
            absolute_strain += torch.sum(torch.abs(labels) * mask).item() / torch.sum(mask).item()
            mean_loss_wo_net += torch.sum(torch.abs(strain - labels) * mask).item() / torch.sum(mask).item()

            # 反向传播和优化
            mean_loss += loss.item()

            mean_loss_epe += torch.sum(torch.sum((strain_pre.detach() - labels) ** 2 * mask_value * mask, dim=1).sqrt()).item() / torch.sum(mask* mask_value).item()
            mean_loss_wo_net_epe += torch.sum(torch.sum((strain.detach() - labels) ** 2 * mask_value * mask, dim=1).sqrt()).item() / torch.sum(mask* mask_value).item()

            # pred_disp = fine_disp + raw_disp.unsqueeze(0)
            disp_mean_loss_rec_valid += torch.sum(torch.abs(pred_disp.detach() - disp_label) * mask.to(
                device)).item() / torch.sum(mask).item()
            disp_mean_epe_valid += torch.sum(
                torch.sum((pred_disp.detach() - disp_label) ** 2 * mask,
                          dim=1).sqrt()).item() / torch.sum(mask).item()
        mean_loss = mean_loss / j
        mean_loss_wo_net = mean_loss_wo_net / j
        absolute_strain = absolute_strain / j / 4
        mean_loss_epe = mean_loss_epe / j
        mean_loss_wo_net_epe = mean_loss_wo_net_epe / j
        disp_mean_loss_rec_valid = disp_mean_loss_rec_valid / j / 2
        disp_mean_epe_valid = disp_mean_epe_valid / j
        # disp_mean_epe_srefine_valid = disp_mean_epe_srefine_valid / j
        print("Disp Total MAE & MEPE: ", disp_mean_loss_rec, disp_mean_epe, disp_mean_loss_rec_valid, disp_mean_epe_valid)
        # print("Disp SRefine MEPE: ", disp_mean_epe_srefine, disp_mean_epe_srefine_valid)
        print("STRAIN Total MAE & MEPE: ", mean_train_loss, mean_train_loss_epe, mean_loss, mean_loss_epe)
        print("STRAIN WON Total MAE & MEPE: ", mean_train_loss_wo_net, mean_train_loss_wo_net_epe, mean_loss_wo_net, mean_loss_wo_net_epe)


def train_DifferNet_FIT_pret(device='cuda', savedir='params', num_epochs=10, dataset_dir=r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\\', img_size=(768, 768)):
    train_dataloader = GenerateDataloader(dataset_dir, type='Train', batch_size=1, shift=True, mode='strain', training_size=img_size[0])
    valid_dataloader = GenerateDataloader(dataset_dir, type='Valid', batch_size=1, shift=True, mode='strain', training_size=img_size[0])

    model_GM = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   inchannel=3,
                   ).to(device)
    model_GM.load_state_dict(torch.load(os.path.join(savedir, 'GMFlowNet_Best.pth')))
    model_GM.eval()
    model_SP = SubpixelCorrNet().to(device)
    model_SP.load_state_dict(torch.load(os.path.join(savedir, 'SubpixelCorrNet_Best.pth')))
    model_SP.eval()
    my_interpolator = interpolator(device=device)

    model = DifferNet_FIT(radius=12, skip=4).to(device).eval()
    if os.path.exists(savedir + '//DifferNet_FIT_Last.pth'):
        model.load_state_dict(torch.load(savedir + '//DifferNet_FIT_Last.pth'))
        # max_loss = np.min(np.array(loss_rec['valid loss']))
        print('Load Parameters: ', savedir + '//DifferNet_FIT_Last.pth')
        max_loss = 0.06
        steps = 0
    else:
        max_loss = 28
        steps = 0

    # kernel_size = 55
    # for kernel_size in kernel_sizes:
    kernel_size = 55
    my_raw_disp_blur = tf.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=kernel_size//2+1)
    my_fine_disp_blur = tf.GaussianBlur(kernel_size=(3, 3), sigma=2)
    grid_x, grid_y = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
    grid_x, grid_y = grid_x.to(device), grid_y.to(device)

    num_epochs -= steps

    steps = 0
    num_epochs -= steps
    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    # 训练模型
    for n in range(num_epochs):
        epoch = n + steps
        mean_train_loss = 0
        mean_train_loss_epe = 0
        mean_train_loss_wo_net = 0
        mean_train_loss_wo_net_epe = 0
        disp_mean_loss_rec = 0
        disp_mean_epe = 0
        for i, (images, labels, mask, fname) in enumerate(tqdm(train_dataloader, desc="Training", unit="iteration")):
            # if n == 0:
            #     continue
            labels, disp_label = labels
            labels = labels.to(device)
            disp_label = disp_label.to(device)
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            # 前向传播
            pred_flow = model_GM(ref_imgs, def_imgs,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1])
            mask = mask.to(device)
            mask[:, :, -20:, :] = False
            mask[:, :, :20, :] = False
            mask[:, :, :, -20:] = False
            mask[:, :, :, :20] = False

            raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
            raw_disp = my_raw_disp_blur(raw_disp)

            # 插值获得中间构型
            u_pos = grid_x + raw_disp[1]
            v_pos = grid_y + raw_disp[0]

            img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

            # 精细像素位移计算
            imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
            fine_disp = model_SP(imgs_array)
            # 插值获得精细大像素位移
            [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                u_pos=grid_x + fine_disp.detach()[0][1],
                v_pos=grid_y + fine_disp.detach()[0][0],
                gray_array_list=[raw_disp.detach()[0], raw_disp.detach()[1]],
                img_mode=False,
                kernel='bspline'
            )
            fine_disp = my_fine_disp_blur(fine_disp[0, :, :, :].detach())
            d_u_fine__d_x_0 = model(fine_disp.unsqueeze(0), with_net=False)
            d_u_raw__d_x_raw = model(raw_disp.unsqueeze(0), with_net=False)
            strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
            u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
            mask_value = torch.abs(u_raw_inter.detach() - disp_label) < 5.0
            mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
            pred_disp = fine_disp + u_raw_inter
            strain_res_pre = model(imgs_array, with_net=True)
            strain_pre = strain + strain_res_pre
            vmin = torch.min(labels[:, :, 2:-2, 2:-2]) - 1.0
            vmax = torch.max(labels[:, :, 2:-2, 2:-2]) + 1.0
            mask_value = (labels < vmax) * (labels > vmin)
            loss = torch.sum(torch.abs(strain_pre - labels) * mask_value * mask) / torch.sum(mask * mask_value) #+ torch.mean(torch.abs(disp_srefine - disp_label) * mask) * 1e-2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += torch.sum(torch.abs(strain_pre - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()
            mean_train_loss_wo_net += torch.sum(torch.abs(strain - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()

            mean_train_loss_epe += torch.sum(torch.sum((strain_pre.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_train_loss_wo_net_epe += torch.sum(torch.sum((strain.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

            # pred_disp = fine_disp + raw_disp.unsqueeze(0)
            disp_mean_loss_rec += torch.sum(torch.abs(pred_disp.detach() - disp_label) * mask).item() / torch.sum(mask).item()
            disp_mean_epe += torch.sum(torch.sum((pred_disp.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            # disp_mean_epe_srefine += torch.sum(torch.sum((disp_srefine.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

        mean_train_loss = mean_train_loss / i
        mean_train_loss_wo_net = mean_train_loss_wo_net / i
        mean_train_loss_epe = mean_train_loss_epe / i
        mean_train_loss_wo_net_epe = mean_train_loss_wo_net_epe / i
        disp_mean_epe = disp_mean_epe / i
        disp_mean_loss_rec = disp_mean_loss_rec / i / 2

        torch.save(model.state_dict(), savedir + '//DifferNet_FIT_Last.pth')
        if epoch % 2 == 0:
            # evaluator.show_seg_img()
            mean_loss = 0
            mean_loss_epe = 0
            absolute_strain = 0
            mean_loss_wo_net = 0
            mean_loss_wo_net_epe = 0
            disp_mean_loss_rec_valid = 0
            disp_mean_epe_valid = 0
            # disp_mean_epe_srefine_valid = 0
            for j, (images, labels, mask, fname) in enumerate(tqdm(valid_dataloader, desc="Validation", unit="iteration")):
                labels, disp_label = labels
                ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                # 前向传播
                pred_flow = model_GM(ref_imgs, def_imgs,
                                     attn_splits_list=[2],
                                     corr_radius_list=[-1],
                                     prop_radius_list=[-1])
                mask = mask.to(device)
                labels = labels.to(device)
                disp_label = disp_label.to(device)
                mask[:, :, -20:, :] = False
                mask[:, :, :20, :] = False
                mask[:, :, :, -20:] = False
                mask[:, :, :, :20] = False

                raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
                raw_disp = my_raw_disp_blur(raw_disp)

                # 插值获得中间构型
                u_pos = grid_x + raw_disp[1]
                v_pos = grid_y + raw_disp[0]
                img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

                # 精细像素位移计算
                imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
                fine_disp = model_SP(imgs_array)
                # 插值获得精细大像素位移
                [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                    u_pos=grid_x + fine_disp[0][1],
                    v_pos=grid_y + fine_disp[0][0],
                    gray_array_list=[raw_disp[0], raw_disp[1]],
                    img_mode=False,
                    kernel='bspline'
                )

                fine_disp = my_fine_disp_blur(fine_disp[0, :, :, :].detach())
                d_u_fine__d_x_0 = model(fine_disp.unsqueeze(0), with_net=False)
                d_u_raw__d_x_raw = model(raw_disp.unsqueeze(0), with_net=False)
                strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
                u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
                mask_value = torch.abs(u_raw_inter.detach() - disp_label) < 5.0
                mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
                pred_disp = fine_disp + u_raw_inter
                strain_res_pre = model(pred_disp, with_net=True)
                strain_pre = strain + strain_res_pre

                vmin = torch.min(labels[:, :, 2:-2, 2:-2]) - 1.0
                vmax = torch.max(labels[:, :, 2:-2, 2:-2]) + 1.0
                mask_value = (labels < vmax) * (labels > vmin)
                loss = torch.sum(torch.abs(strain_pre - labels) * mask_value * mask) / torch.sum(mask * mask_value)
                absolute_strain += torch.sum(torch.abs(labels) * mask).item() / torch.sum(mask).item()
                mean_loss_wo_net += torch.sum(torch.abs(strain - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()

                # 反向传播和优化
                mean_loss += loss.item()
                plt.imshow()

                mean_loss_epe += torch.sum(torch.sum((strain_pre.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
                mean_loss_wo_net_epe += torch.sum(torch.sum((strain.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

                # pred_disp = fine_disp + raw_disp.unsqueeze(0)
                disp_mean_loss_rec_valid += torch.sum(torch.abs(pred_disp.detach() - disp_label) * mask.to(
                    device)).item() / torch.sum(mask).item()
                disp_mean_epe_valid += torch.sum(
                    torch.sum((pred_disp.detach() - disp_label) ** 2 * mask,
                              dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_loss = mean_loss / j
            mean_loss_wo_net = mean_loss_wo_net / j
            absolute_strain = absolute_strain / j / 4
            mean_loss_epe = mean_loss_epe / j
            mean_loss_wo_net_epe = mean_loss_wo_net_epe / j
            disp_mean_loss_rec_valid = disp_mean_loss_rec_valid / j / 2
            disp_mean_epe_valid = disp_mean_epe_valid / j
            # disp_mean_epe_srefine_valid = disp_mean_epe_srefine_valid / j
            print("Disp Total MAE & MEPE: ", disp_mean_loss_rec, disp_mean_epe, disp_mean_loss_rec_valid, disp_mean_epe_valid)
            # print("Disp SRefine MEPE: ", disp_mean_epe_srefine, disp_mean_epe_srefine_valid)
            print("STRAIN Total MAE & MEPE: ", mean_train_loss, mean_train_loss_epe, mean_loss, mean_loss_epe)
            print("STRAIN WON Total MAE & MEPE: ", mean_train_loss_wo_net, mean_train_loss_wo_net_epe, mean_loss_wo_net, mean_loss_wo_net_epe)
            # # 打印训练信息
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train won Loss: {:.4f}, Valid Loss: {:.4f}, Valid won Loss: {:.4f}, Valid Absolute Strain: {:.4f}'.
                  format(epoch + 1, num_epochs + steps, mean_train_loss, mean_train_loss_wo_net,
                         mean_loss, mean_loss_wo_net, absolute_strain))
        #
            if mean_loss < max_loss:
                torch.save(model.state_dict(), savedir + '//DifferNet_FIT_Best.pth')
                max_loss = mean_loss
            torch.save(model.state_dict(), savedir + '//DifferNet_FIT_Last.pth')


def train_DifferNet_FIT_inter(device='cuda', savedir='params', num_epochs=100, dataset_dir=r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\\', img_size=(768, 768)):
    train_dataloader = GenerateDataloader(dataset_dir, type='Train', batch_size=1, shift=True, mode='strain', training_size=img_size[0])
    valid_dataloader = GenerateDataloader(dataset_dir, type='Valid', batch_size=1, shift=True, mode='strain', training_size=img_size[0])

    model_GM = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   inchannel=3,
                   ).to(device)
    model_GM.load_state_dict(torch.load(os.path.join(savedir, 'GMFlowNet_Best.pth')))
    model_GM.eval()
    model_SP = SubpixelCorrNet().to(device)
    model_SP.load_state_dict(torch.load(os.path.join(savedir, 'SubpixelCorrNet_Best.pth')))
    model_SP.eval()
    my_interpolator = interpolator(device=device)

    model = DifferNet_FIT_Inter(radius=12, skip=4).to(device).eval()
    if os.path.exists(savedir + '//DifferNet_INT_Last.pth'):
        model.load_state_dict(torch.load(savedir + '//DifferNet_INT_Last.pth'))
        # max_loss = np.min(np.array(loss_rec['valid loss']))
        print('Load Parameters: ', savedir + '//DifferNet_INT_Last.pth')
        max_loss = 0.06
        steps = 0
    else:
        max_loss = 28
        steps = 0

    # kernel_size = 55
    # for kernel_size in kernel_sizes:
    kernel_size = 55
    my_raw_disp_blur = tf.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=kernel_size//2+1)
    my_fine_disp_blur = tf.GaussianBlur(kernel_size=(25, 25), sigma=12)
    my_strain_blur = tf.GaussianBlur(kernel_size=(25, 25), sigma=15)
    my_strain_blur_2 = tf.GaussianBlur(kernel_size=(23, 23), sigma=15)

    grid_x, grid_y = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
    grid_x, grid_y = grid_x.to(device), grid_y.to(device)

    num_epochs -= steps

    steps = 0
    num_epochs -= steps
    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    num_iter = 2
    # 训练模型
    for n in range(num_epochs):
        epoch = n + steps
        mean_train_loss = 0
        mean_train_loss_epe = 0
        mean_train_loss_wo_net = 0
        mean_train_loss_wo_net_epe = 0
        disp_mean_loss_rec = 0
        disp_mean_epe = 0
        for i, (images, labels, mask, fname) in enumerate(tqdm(train_dataloader, desc="Training", unit="iteration")):
            # if n == 0:
            #     continue
            labels, disp_label = labels
            labels = labels.to(device)
            disp_label = disp_label.to(device)
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            # 前向传播
            pred_flow = model_GM(ref_imgs, def_imgs,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1])
            mask = mask.to(device)
            mask[:, :, -20:, :] = False
            mask[:, :, :20, :] = False
            mask[:, :, :, -20:] = False
            mask[:, :, :, :20] = False

            raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
            raw_disp = my_raw_disp_blur(raw_disp)

            tmp_v_raw = raw_disp[1]
            tmp_u_raw = raw_disp[0]

            for j in range(num_iter):

                u_pos = grid_x + tmp_v_raw
                v_pos = grid_y + tmp_u_raw

                tmp_cur_img = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

                # 精细像素位移计算
                tmp_fine_disp = model_SP(torch.stack([ref_imgs[0, 0, :, :], tmp_cur_img], dim=0).unsqueeze(0).cuda())[0, :, :, :].detach()

                # 插值获得精细大像素位移

                if j == num_iter - 1:
                    tmp_fine_disp_save = copy.deepcopy(tmp_fine_disp)

                tmp_fine_disp = my_fine_disp_blur(tmp_fine_disp)
                [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                    u_pos=grid_x + tmp_fine_disp[1],
                    v_pos=grid_y + tmp_fine_disp[0],
                    gray_array_list=[tmp_u_raw, tmp_v_raw],
                    img_mode=False,
                    kernel='bspline'
                )
                tmp_u_raw = tmp_fine_disp[0] + u_raw_inter
                tmp_v_raw = tmp_fine_disp[1] + v_raw_inter
                raw_disp = my_raw_disp_blur(torch.stack([tmp_u_raw, tmp_v_raw], dim=0))
                tmp_u_raw, tmp_v_raw = raw_disp[0], raw_disp[1]
                if j == num_iter - 1:
                    raw_disp = torch.stack([u_raw_inter, v_raw_inter], dim=0)

            #     plt.imshow(tmp_cur_img.cpu().numpy())
            #     plt.show()
            # plt.imshow(ref_imgs[0, 0, :, :].cpu().numpy())
            # plt.show()

            # d_u_fine__d_x_0 = model(tmp_fine_disp, with_net=False)
            # d_u_raw__d_x_raw = model(raw_disp.unsqueeze(0), with_net=False)
            # strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
            raw_disp = raw_disp.unsqueeze(0)
            mask_value = torch.abs(raw_disp.detach() - disp_label) < 5.0
            mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
            pred_disp = tmp_fine_disp_save + raw_disp
            strain = model(pred_disp, with_net=False)
            strain = my_strain_blur(strain[0, :, :, :])
            strain = my_strain_blur_2(strain).unsqueeze(0)
            # plt.imshow((mask * strain)[0, 0, :, :].cpu().numpy())
            # plt.show()
            # plt.imshow((mask * labels)[0, 0, :, :].cpu().numpy())
            # plt.show()
            strain_res_pre = model(torch.stack([ref_imgs[0, 0, :, :], tmp_cur_img], dim=0).unsqueeze(0).cuda(), with_net=True)
            strain_pre = strain + strain_res_pre
            vmin = torch.min(labels[:, :, 2:-2, 2:-2]) - 0.8
            vmax = torch.max(labels[:, :, 2:-2, 2:-2]) + 0.8
            mask_value = (labels < vmax) * (labels > vmin)
            loss = torch.sum(torch.abs(strain_pre - labels) * mask_value * mask) / torch.sum(mask * mask_value) #+ torch.mean(torch.abs(disp_srefine - disp_label) * mask) * 1e-2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += torch.sum(torch.abs(strain_pre - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()
            mean_train_loss_wo_net += torch.sum(torch.abs(strain - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()

            mean_train_loss_epe += torch.sum(torch.sum((strain_pre.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_train_loss_wo_net_epe += torch.sum(torch.sum((strain.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

            # pred_disp = fine_disp + raw_disp.unsqueeze(0)
            disp_mean_loss_rec += torch.sum(torch.abs(pred_disp.detach() - disp_label) * mask).item() / torch.sum(mask).item()
            disp_mean_epe += torch.sum(torch.sum((pred_disp.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            # disp_mean_epe_srefine += torch.sum(torch.sum((disp_srefine.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

        mean_train_loss = mean_train_loss / i
        mean_train_loss_wo_net = mean_train_loss_wo_net / i
        mean_train_loss_epe = mean_train_loss_epe / i
        mean_train_loss_wo_net_epe = mean_train_loss_wo_net_epe / i
        disp_mean_epe = disp_mean_epe / i
        disp_mean_loss_rec = disp_mean_loss_rec / i / 2
        print(mean_train_loss, mean_train_loss_wo_net)

        torch.save(model.state_dict(), savedir + '//DifferNet_INT_Last.pth')
        if epoch % 3 == 0:
            # evaluator.show_seg_img()
            mean_loss = 0
            mean_loss_epe = 0
            absolute_strain = 0
            mean_loss_wo_net = 0
            mean_loss_wo_net_epe = 0
            disp_mean_loss_rec_valid = 0
            disp_mean_epe_valid = 0
            # disp_mean_epe_srefine_valid = 0
            for j, (images, labels, mask, fname) in enumerate(tqdm(valid_dataloader, desc="Validation", unit="iteration")):
                labels, disp_label = labels
                ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                # 前向传播
                pred_flow = model_GM(ref_imgs, def_imgs,
                                     attn_splits_list=[2],
                                     corr_radius_list=[-1],
                                     prop_radius_list=[-1])
                mask = mask.to(device)
                labels = labels.to(device)
                disp_label = disp_label.to(device)
                mask[:, :, -20:, :] = False
                mask[:, :, :20, :] = False
                mask[:, :, :, -20:] = False
                mask[:, :, :, :20] = False

                raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
                raw_disp = my_raw_disp_blur(raw_disp)

                # 插值获得中间构型
                u_pos = grid_x + raw_disp[1]
                v_pos = grid_y + raw_disp[0]
                img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

                # 精细像素位移计算
                imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
                fine_disp = model_SP(imgs_array)
                # 插值获得精细大像素位移
                fine_disp = my_fine_disp_blur(fine_disp[0, :, :]).unsqueeze(0)
                [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                    u_pos=grid_x + fine_disp[0][1],
                    v_pos=grid_y + fine_disp[0][0],
                    gray_array_list=[raw_disp[0], raw_disp[1]],
                    img_mode=False,
                    kernel='bspline'
                )

                d_u_fine__d_x_0 = model(fine_disp, with_net=False)
                d_u_raw__d_x_raw = model(raw_disp.unsqueeze(0), with_net=False)
                strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
                strain = my_strain_blur(strain[0, :, :, :])
                strain = my_strain_blur_2(strain).unsqueeze(0)
                u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
                mask_value = torch.abs(u_raw_inter.detach() - disp_label) < 5.0
                mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
                pred_disp = fine_disp + u_raw_inter
                strain_res_pre = model(imgs_array, with_net=True)
                strain_pre = strain + strain_res_pre

                vmin = torch.min(labels[:, :, 2:-2, 2:-2]) - 1.0
                vmax = torch.max(labels[:, :, 2:-2, 2:-2]) + 1.0
                mask_value = (labels < vmax) * (labels > vmin)
                absolute_strain += torch.sum(torch.abs(labels) * mask).item() / torch.sum(mask).item()

                mean_loss += torch.sum(torch.abs(strain_pre.detach() - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()
                mean_loss_wo_net += torch.sum(torch.abs(strain - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()

                mean_loss_epe += torch.sum(torch.sum((strain_pre.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
                mean_loss_wo_net_epe += torch.sum(torch.sum((strain.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

                # pred_disp = fine_disp + raw_disp.unsqueeze(0)
                disp_mean_loss_rec_valid += torch.sum(torch.abs(pred_disp.detach() - disp_label) * mask).item() / torch.sum(mask).item()
                disp_mean_epe_valid += torch.sum(torch.sum((pred_disp.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_loss = mean_loss / j
            mean_loss_wo_net = mean_loss_wo_net / j
            absolute_strain = absolute_strain / j / 4
            mean_loss_epe = mean_loss_epe / j
            mean_loss_wo_net_epe = mean_loss_wo_net_epe / j
            disp_mean_loss_rec_valid = disp_mean_loss_rec_valid / j / 2
            disp_mean_epe_valid = disp_mean_epe_valid / j
            # disp_mean_epe_srefine_valid = disp_mean_epe_srefine_valid / j
            print("Disp Total MAE & MEPE: ", disp_mean_loss_rec, disp_mean_epe, disp_mean_loss_rec_valid, disp_mean_epe_valid)
            print("STRAIN Total MAE & MEPE: ", mean_train_loss, mean_train_loss_epe, mean_loss, mean_loss_epe)
            print("STRAIN WON Total MAE & MEPE: ", mean_train_loss_wo_net, mean_train_loss_wo_net_epe, mean_loss_wo_net, mean_loss_wo_net_epe)
            # # 打印训练信息
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train won Loss: {:.4f}, Valid Loss: {:.4f}, Valid won Loss: {:.4f}, Valid Absolute Strain: {:.4f}'.
                  format(epoch + 1, num_epochs + steps, mean_train_loss, mean_train_loss_wo_net,
                         mean_loss, mean_loss_wo_net, absolute_strain))
        #
            if mean_loss < max_loss:
                torch.save(model.state_dict(), savedir + '//DifferNet_INT_Best.pth')
                max_loss = mean_loss
            torch.save(model.state_dict(), savedir + '//DifferNet_INT_Last.pth')

def train_DifferNet_FIT_inter2(device='cuda', savedir='params', num_epochs=100, dataset_dir=r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\\', img_size=(768, 768)):
    train_dataloader = GenerateDataloader(dataset_dir, type='Train', batch_size=1, shift=True, mode='strain', training_size=img_size[0])
    valid_dataloader = GenerateDataloader(dataset_dir, type='Valid', batch_size=1, shift=True, mode='strain', training_size=img_size[0])

    model_GM = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   inchannel=3,
                   ).to(device)
    model_GM.load_state_dict(torch.load(os.path.join(savedir, 'GMFlowNet_Best.pth')))
    model_GM.eval()
    model_SP = SubpixelCorrNet().to(device)
    model_SP.load_state_dict(torch.load(os.path.join(savedir, 'SubpixelCorrNet_Best.pth')))
    model_SP.eval()
    my_interpolator = interpolator(device=device)

    model = DifferNet_FIT_Inter(radius=12, skip=4).to(device).eval()
    if os.path.exists(savedir + '//DifferNet_INT_Last.pth'):
        model.load_state_dict(torch.load(savedir + '//DifferNet_INT_Last.pth'))
        # max_loss = np.min(np.array(loss_rec['valid loss']))
        print('Load Parameters: ', savedir + '//DifferNet_INT_Last.pth')
        max_loss = 0.06
        steps = 0
    else:
        max_loss = 28
        steps = 0

    # kernel_size = 55
    # for kernel_size in kernel_sizes:
    kernel_size = 55
    my_raw_disp_blur = tf.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=kernel_size//2+1)
    my_fine_disp_blur = tf.GaussianBlur(kernel_size=(25, 25), sigma=12)
    my_strain_blur = tf.GaussianBlur(kernel_size=(25, 25), sigma=15)
    my_strain_blur_2 = tf.GaussianBlur(kernel_size=(23, 23), sigma=15)

    grid_x, grid_y = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
    grid_x, grid_y = grid_x.to(device), grid_y.to(device)

    num_epochs -= steps

    steps = 0
    num_epochs -= steps
    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    num_iter = 2
    # 训练模型
    for n in range(num_epochs):
        epoch = n + steps
        mean_train_loss = 0
        mean_train_loss_epe = 0
        mean_train_loss_wo_net = 0
        mean_train_loss_wo_net_epe = 0
        disp_mean_loss_rec = 0
        disp_mean_epe = 0
        for i, (images, labels, mask, fname) in enumerate(tqdm(train_dataloader, desc="Training", unit="iteration")):
            # if n == 0:
            #     continue
            labels, disp_label = labels
            labels = labels.to(device)
            disp_label = disp_label.to(device)
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            # 前向传播
            pred_flow = model_GM(ref_imgs, def_imgs,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1])
            mask = mask.to(device)
            mask[:, :, -20:, :] = False
            mask[:, :, :20, :] = False
            mask[:, :, :, -20:] = False
            mask[:, :, :, :20] = False

            raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
            raw_disp = my_raw_disp_blur(raw_disp)

            tmp_v_raw = raw_disp[1]
            tmp_u_raw = raw_disp[0]

            for l in range(num_iter):

                u_pos = grid_x + tmp_v_raw
                v_pos = grid_y + tmp_u_raw

                tmp_cur_img = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

                # 精细像素位移计算
                tmp_fine_disp = model_SP(torch.stack([ref_imgs[0, 0, :, :], tmp_cur_img], dim=0).unsqueeze(0).cuda())[0, :, :, :].detach()

                # 插值获得精细大像素位移

                if l == num_iter - 1:
                    tmp_fine_disp_save = copy.deepcopy(tmp_fine_disp)

                tmp_fine_disp = my_fine_disp_blur(tmp_fine_disp)
                [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                    u_pos=grid_x + tmp_fine_disp[1],
                    v_pos=grid_y + tmp_fine_disp[0],
                    gray_array_list=[tmp_u_raw, tmp_v_raw],
                    img_mode=False,
                    kernel='bspline'
                )
                tmp_u_raw = tmp_fine_disp[0] + u_raw_inter
                tmp_v_raw = tmp_fine_disp[1] + v_raw_inter
                raw_disp = my_raw_disp_blur(torch.stack([tmp_u_raw, tmp_v_raw], dim=0))
                tmp_u_raw, tmp_v_raw = raw_disp[0], raw_disp[1]
                if l == num_iter - 1:
                    raw_disp = torch.stack([u_raw_inter, v_raw_inter], dim=0)

            #     plt.imshow(tmp_cur_img.cpu().numpy())
            #     plt.show()
            # plt.imshow(ref_imgs[0, 0, :, :].cpu().numpy())
            # plt.show()

            # d_u_fine__d_x_0 = model(tmp_fine_disp, with_net=False)
            # d_u_raw__d_x_raw = model(raw_disp.unsqueeze(0), with_net=False)
            # strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
            raw_disp = raw_disp.unsqueeze(0)
            mask_value = torch.abs(raw_disp.detach() - disp_label) < 5.0
            mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
            pred_disp = tmp_fine_disp_save + raw_disp
            strain = model(pred_disp, with_net=False)
            strain = my_strain_blur(strain[0, :, :, :])
            strain = my_strain_blur_2(strain).unsqueeze(0)
            # plt.imshow((mask * strain)[0, 0, :, :].cpu().numpy())
            # plt.show()
            # plt.imshow((mask * labels)[0, 0, :, :].cpu().numpy())
            # plt.show()
            strain_res_pre = model(torch.stack([ref_imgs[0, 0, :, :], tmp_cur_img], dim=0).unsqueeze(0).cuda(), with_net=True)
            strain_pre = strain + strain_res_pre
            vmin = torch.min(labels[:, :, 2:-2, 2:-2]) - 0.8
            vmax = torch.max(labels[:, :, 2:-2, 2:-2]) + 0.8
            mask_value = (labels < vmax) * (labels > vmin)
            loss = torch.sum(torch.abs(strain_pre - labels) * mask_value * mask) / torch.sum(mask * mask_value) #+ torch.mean(torch.abs(disp_srefine - disp_label) * mask) * 1e-2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += torch.sum(torch.abs(strain_pre - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()
            mean_train_loss_wo_net += torch.sum(torch.abs(strain - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()

            mean_train_loss_epe += torch.sum(torch.sum((strain_pre.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_train_loss_wo_net_epe += torch.sum(torch.sum((strain.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

            # pred_disp = fine_disp + raw_disp.unsqueeze(0)
            disp_mean_loss_rec += torch.sum(torch.abs(pred_disp.detach() - disp_label) * mask).item() / torch.sum(mask).item()
            disp_mean_epe += torch.sum(torch.sum((pred_disp.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            # disp_mean_epe_srefine += torch.sum(torch.sum((disp_srefine.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

        mean_train_loss = mean_train_loss / i
        mean_train_loss_wo_net = mean_train_loss_wo_net / i
        mean_train_loss_epe = mean_train_loss_epe / i
        mean_train_loss_wo_net_epe = mean_train_loss_wo_net_epe / i
        disp_mean_epe = disp_mean_epe / i
        disp_mean_loss_rec = disp_mean_loss_rec / i / 2
        print(mean_train_loss, mean_train_loss_wo_net)

        torch.save(model.state_dict(), savedir + '//DifferNet_INT_Last.pth')
        if epoch % 3 == 0:
            # evaluator.show_seg_img()
            mean_loss = 0
            mean_loss_epe = 0
            absolute_strain = 0
            mean_loss_wo_net = 0
            mean_loss_wo_net_epe = 0
            disp_mean_loss_rec_valid = 0
            disp_mean_epe_valid = 0
            # disp_mean_epe_srefine_valid = 0
            for j, (images, labels, mask, fname) in enumerate(tqdm(valid_dataloader, desc="Validation", unit="iteration")):
                labels, disp_label = labels
                ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                # 前向传播
                pred_flow = model_GM(ref_imgs, def_imgs,
                                     attn_splits_list=[2],
                                     corr_radius_list=[-1],
                                     prop_radius_list=[-1])
                mask = mask.to(device)
                labels = labels.to(device)
                disp_label = disp_label.to(device)
                mask[:, :, -20:, :] = False
                mask[:, :, :20, :] = False
                mask[:, :, :, -20:] = False
                mask[:, :, :, :20] = False

                raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
                raw_disp = my_raw_disp_blur(raw_disp)

                tmp_v_raw = raw_disp[1]
                tmp_u_raw = raw_disp[0]

                for k in range(num_iter):

                    u_pos = grid_x + tmp_v_raw
                    v_pos = grid_y + tmp_u_raw

                    tmp_cur_img = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

                    # 精细像素位移计算
                    tmp_fine_disp = model_SP(
                        torch.stack([ref_imgs[0, 0, :, :], tmp_cur_img], dim=0).unsqueeze(0).cuda())[0, :, :,
                                    :].detach()

                    # 插值获得精细大像素位移

                    if k == num_iter - 1:
                        tmp_fine_disp_save = copy.deepcopy(tmp_fine_disp)

                    tmp_fine_disp = my_fine_disp_blur(tmp_fine_disp)
                    [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                        u_pos=grid_x + tmp_fine_disp[1],
                        v_pos=grid_y + tmp_fine_disp[0],
                        gray_array_list=[tmp_u_raw, tmp_v_raw],
                        img_mode=False,
                        kernel='bspline'
                    )
                    tmp_u_raw = tmp_fine_disp[0] + u_raw_inter
                    tmp_v_raw = tmp_fine_disp[1] + v_raw_inter
                    raw_disp = my_raw_disp_blur(torch.stack([tmp_u_raw, tmp_v_raw], dim=0))
                    tmp_u_raw, tmp_v_raw = raw_disp[0], raw_disp[1]
                    if k == num_iter - 1:
                        raw_disp = torch.stack([u_raw_inter, v_raw_inter], dim=0)

                #     plt.imshow(tmp_cur_img.cpu().numpy())
                #     plt.show()
                # plt.imshow(ref_imgs[0, 0, :, :].cpu().numpy())
                # plt.show()

                # d_u_fine__d_x_0 = model(tmp_fine_disp, with_net=False)
                # d_u_raw__d_x_raw = model(raw_disp.unsqueeze(0), with_net=False)
                # strain = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
                raw_disp = raw_disp.unsqueeze(0)
                mask_value = torch.abs(raw_disp.detach() - disp_label) < 5.0
                mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
                pred_disp = tmp_fine_disp_save + raw_disp
                strain = model(pred_disp, with_net=False)
                strain = my_strain_blur(strain[0, :, :, :])
                strain = my_strain_blur_2(strain).unsqueeze(0)
                # plt.imshow((mask * strain)[0, 0, :, :].cpu().numpy())
                # plt.show()
                # plt.imshow((mask * labels)[0, 0, :, :].cpu().numpy())
                # plt.show()
                strain_res_pre = model(torch.stack([ref_imgs[0, 0, :, :], tmp_cur_img], dim=0).unsqueeze(0).cuda(),
                                       with_net=True)
                strain_pre = strain + strain_res_pre
                vmin = torch.min(labels[:, :, 2:-2, 2:-2]) - 0.8
                vmax = torch.max(labels[:, :, 2:-2, 2:-2]) + 0.8

                mask_value = (labels < vmax) * (labels > vmin)
                absolute_strain += torch.sum(torch.abs(labels) * mask).item() / torch.sum(mask).item()

                mean_loss += torch.sum(torch.abs(strain_pre.detach() - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()
                mean_loss_wo_net += torch.sum(torch.abs(strain - labels) * mask_value * mask).item() / torch.sum(mask * mask_value).item()

                mean_loss_epe += torch.sum(torch.sum((strain_pre.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
                mean_loss_wo_net_epe += torch.sum(torch.sum((strain.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()

                # pred_disp = fine_disp + raw_disp.unsqueeze(0)
                disp_mean_loss_rec_valid += torch.sum(torch.abs(pred_disp.detach() - disp_label) * mask).item() / torch.sum(mask).item()
                disp_mean_epe_valid += torch.sum(torch.sum((pred_disp.detach() - disp_label) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_loss = mean_loss / j
            mean_loss_wo_net = mean_loss_wo_net / j
            absolute_strain = absolute_strain / j / 4
            mean_loss_epe = mean_loss_epe / j
            mean_loss_wo_net_epe = mean_loss_wo_net_epe / j
            disp_mean_loss_rec_valid = disp_mean_loss_rec_valid / j / 2
            disp_mean_epe_valid = disp_mean_epe_valid / j
            # disp_mean_epe_srefine_valid = disp_mean_epe_srefine_valid / j
            print("Disp Total MAE & MEPE: ", disp_mean_loss_rec, disp_mean_epe, disp_mean_loss_rec_valid, disp_mean_epe_valid)
            print("STRAIN Total MAE & MEPE: ", mean_train_loss, mean_train_loss_epe, mean_loss, mean_loss_epe)
            print("STRAIN WON Total MAE & MEPE: ", mean_train_loss_wo_net, mean_train_loss_wo_net_epe, mean_loss_wo_net, mean_loss_wo_net_epe)
            # # 打印训练信息
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train won Loss: {:.4f}, Valid Loss: {:.4f}, Valid won Loss: {:.4f}, Valid Absolute Strain: {:.4f}'.
                  format(epoch + 1, num_epochs + steps, mean_train_loss, mean_train_loss_wo_net,
                         mean_loss, mean_loss_wo_net, absolute_strain))
        #
            if mean_loss < max_loss:
                torch.save(model.state_dict(), savedir + '//DifferNet_INT_Best.pth')
                max_loss = mean_loss
            torch.save(model.state_dict(), savedir + '//DifferNet_INT_Last.pth')

if __name__ == '__main__':
    dataset_dir = r'E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\\'
    # train_TransUNet_pret(dataset_dir=dataset_dir, scale=2)
    # train_DifferNet_FIT_pret(dataset_dir=dataset_dir)
    train_DifferNet_FIT_inter2(dataset_dir=dataset_dir)
    # train_DifferNet_FIT_eval(dataset_dir=dataset_dir)