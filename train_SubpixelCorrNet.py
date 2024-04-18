
from networks.SubpixelCorrNet import SubpixelCorrNet
from networks.gmflow.gmflow import GMFlow
from utils.dataloader import GenerateDataloader
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.interpolation import interpolator
import torchvision.transforms as tf


def train_SubpixelCorrNet_eval(device='cuda', savedir='params', num_epochs=500, dataset_dir=r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\\', load_init=False, img_size=(768, 768)):
    batch_size = 1
    train_dataloader = GenerateDataloader(dataset_dir, type='Train', batch_size=batch_size, shift=True)
    valid_dataloader = GenerateDataloader(dataset_dir, type='Valid', batch_size=batch_size, shift=True)

    model = SubpixelCorrNet().to(device)
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
    my_interpolator = interpolator(device=device)
    my_raw_disp_blur = tf.GaussianBlur(kernel_size=(35, 35), sigma=13)

    grid_x, grid_y = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
    grid_x, grid_y = grid_x.to(device), grid_y.to(device)

    model.load_state_dict(torch.load(savedir + '//SubpixelCorrNet_Best.pth'))

    model.eval()

    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # 训练模型
    for n in range(num_epochs):
        epoch = n
        mean_train_loss = 0
        train_epe = 0
        mean_train_raw_loss = 0
        train_raw_epe = 0
        for i, (images, labels, mask, fname) in enumerate(tqdm(train_dataloader, desc="Training", unit="iteration")):
            # if n == 0:
            #     continue
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            labels = labels.to(device)
            mask = mask.to(device)
            mask[:, :, -20:, :] = False
            mask[:, :, :20, :] = False
            mask[:, :, :, -20:] = False
            mask[:, :, :, :20] = False
            # 前向传播
            pred_flow = model_GM(ref_imgs, def_imgs,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1])

            raw_disp_init = pred_flow['flow_preds'][-1][0, :, :, :].detach()
            raw_disp = my_raw_disp_blur(raw_disp_init)
            # 插值获得中间构型
            u_pos = grid_x + raw_disp[1]
            v_pos = grid_y + raw_disp[0]

            img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

            # 精细像素位移计算
            imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
            fine_disp = model(imgs_array.to(device))

            [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_bilinear_list(
                u_pos=grid_x + fine_disp.detach()[0][1],
                v_pos=grid_y + fine_disp.detach()[0][0],
                gray_array_list=[raw_disp.detach()[0], raw_disp.detach()[1]],
                img_mode=False,
            )
            u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
            mask_value = torch.abs(u_raw_inter.detach() - labels) < 5.0
            mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)
            loss = torch.sum(torch.abs(fine_disp + u_raw_inter.detach() - labels) * mask) #+ \
                   # 0.2 * torch.sum(torch.abs(raw_disp1 + cat_raw_uv_inter.detach() - labels) * mask) + \
                   # 0.1 * torch.sum(torch.abs(raw_disp2 + cat_raw_uv_inter.detach() - labels) * mask)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_raw_loss += torch.sum(torch.abs(raw_disp_init.detach() - labels) * mask).item() / torch.sum(mask).item()
            train_raw_epe += torch.sum(torch.sum((raw_disp_init.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_train_loss += torch.sum(torch.abs(fine_disp + u_raw_inter.detach() - labels) * mask).item() / torch.sum(mask).item()
            train_epe += torch.sum(torch.sum((fine_disp + u_raw_inter.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
        mean_train_loss = mean_train_loss / i / 2
        train_epe = train_epe / i
        mean_train_raw_loss = mean_train_raw_loss / i / 2
        train_raw_epe = train_raw_epe / i
        print('Train raw MAE %.5f, Train raw EPE %.5f'%(mean_train_raw_loss, train_raw_epe))
        print('Train MAE %.5f, Train EPE %.5f'%(mean_train_loss, train_epe))

        # evaluator.show_seg_img()
        mean_loss = 0
        valid_epe = 0
        mean_raw_loss = 0
        valid_raw_epe = 0
        model.eval()
        for j, (images, labels, mask, fname) in enumerate(tqdm(valid_dataloader, desc="Validation", unit="iteration")):
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            labels = labels.to(device)
            mask = mask.to(device)
            mask[:, :, -20:, :] = False
            mask[:, :, :20, :] = False
            mask[:, :, :, -20:] = False
            mask[:, :, :, :20] = False
            # 前向传播
            pred_flow = model_GM(ref_imgs, def_imgs,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1])

            raw_disp_init = pred_flow['flow_preds'][-1][0, :, :, :].detach()
            raw_disp = my_raw_disp_blur(raw_disp_init)

            # 插值获得中间构型
            u_pos = grid_x + raw_disp[1]
            v_pos = grid_y + raw_disp[0]

            img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

            # 精细像素位移计算
            imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
            fine_disp = model(imgs_array.to(device))
            # 插值获得精细大像素位移
            [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                u_pos=grid_x + fine_disp.detach()[0][1],
                v_pos=grid_y + fine_disp.detach()[0][0],
                gray_array_list=[raw_disp.detach()[0], raw_disp.detach()[1]],
                img_mode=False,
                kernel='bspline'
            )
            u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
            mask_value = torch.abs(u_raw_inter.detach() - labels) < 5.0
            mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)

            # 插值获得精细大像素位移
            loss = torch.sum(torch.abs(fine_disp + u_raw_inter.detach() - labels) * mask).item() / torch.sum(mask).item()
            # 反向传播和优化
            mean_loss += loss
            valid_epe += torch.sum(torch.sum((fine_disp.detach() + u_raw_inter - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_raw_loss += torch.sum(torch.abs(raw_disp_init.detach() - labels) * mask).item() / torch.sum(mask).item()
            valid_raw_epe += torch.sum(torch.sum((raw_disp_init.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
        mean_loss = mean_loss / j / 2
        valid_epe = valid_epe / j
        mean_raw_loss = mean_raw_loss / j / 2
        valid_raw_epe = valid_raw_epe / j
        # evaluator.show_seg_img()
        # 打印训练信息
        print('Valid raw MAE %.5f, Valid raw EPE %.5f'%(mean_raw_loss, valid_raw_epe))
        print('Valid MAE %.5f, Valid EPE %.5f'%(mean_loss, valid_epe))



def train_SubpixelCorrNet_pret(device='cuda', savedir='params', num_epochs=500, dataset_dir=r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\\', load_init=False, img_size=(768, 768)):
    batch_size = 1
    train_dataloader = GenerateDataloader(dataset_dir, type='Train', batch_size=batch_size, shift=True)
    valid_dataloader = GenerateDataloader(dataset_dir, type='Valid', batch_size=batch_size, shift=True)

    model = SubpixelCorrNet().to(device)
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
    my_interpolator = interpolator(device=device)
    my_raw_disp_blur = tf.GaussianBlur(kernel_size=(35, 35), sigma=13)
    my_fine_disp_blur = tf.GaussianBlur(kernel_size=(7, 7), sigma=4)

    grid_x, grid_y = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
    grid_x, grid_y = grid_x.to(device), grid_y.to(device)

    if not load_init:
        if os.path.exists(savedir + '//SubpixelCorrNet_Last.pth'):
            # steps = 50 * ((loss_rec['train epoch'][-1] + 1) // 50)
            steps = 0
            model.load_state_dict(torch.load(savedir + '//SubpixelCorrNet_Last.pth'))
            # model.load_state_dict(torch.load(savedir + '//SubpixelCorrNet_init.pth'))
            print('load: ', savedir + '//SubpixelCorrNet_Last.pth')
            # max_loss = np.max(np.array(loss_rec['valid loss']))
            max_loss = 0.31
        else:
            max_loss = 28
            steps = 0
    else:
        pre_trained_paramesters = torch.load(savedir + '//SubpixelCorrNet_TN_Best.pth')
        model.load_state_dict(pre_trained_paramesters)
        max_loss = 28
        steps = 0

    num_epochs -= steps
    model.correlator1.requires_grad_(False)
    model.correlator3.requires_grad_(False)

    num_epochs -= steps
    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # 训练模型
    for n in range(num_epochs):
        epoch = n + steps
        mean_train_loss = 0
        train_epe = 0
        model.train()
        for i, (images, labels, mask, fname) in enumerate(tqdm(train_dataloader, desc="Training", unit="iteration")):
            # if n == 0:
            #     continue
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            labels = labels.to(device)
            mask = mask.to(device)
            mask[:, :, -20:, :] = False
            mask[:, :, :20, :] = False
            mask[:, :, :, -20:] = False
            mask[:, :, :, :20] = False
            # 前向传播
            pred_flow = model_GM(ref_imgs, def_imgs,
                                 attn_splits_list=[2],
                                 corr_radius_list=[-1],
                                 prop_radius_list=[-1])

            raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
            raw_disp[:, ] = my_raw_disp_blur(raw_disp)
            # 插值获得中间构型
            u_pos = grid_x + raw_disp[1]
            v_pos = grid_y + raw_disp[0]

            img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

            # 精细像素位移计算
            imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
            fine_disp, raw_disp1, raw_disp2 = model(imgs_array.to(device))
            # 插值获得精细大像素位移
            # [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
            #     u_pos=grid_x + fine_disp.detach()[0][1],
            #     v_pos=grid_y + fine_disp.detach()[0][0],
            #     gray_array_list=[raw_disp.detach()[0], raw_disp.detach()[1]],
            #     img_mode=False,
            #     kernel='bspline'
            # )
            # fine_disp_filter = my_fine_disp_blur(fine_disp)
            [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_bilinear_list(
                u_pos=grid_x + fine_disp.detach()[0][1],
                v_pos=grid_y + fine_disp.detach()[0][0],
                gray_array_list=[raw_disp.detach()[0], raw_disp.detach()[1]],
                img_mode=False,
            )
            u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
            mask_value = torch.abs(u_raw_inter.detach() - labels) < 5.0
            mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)

            # plt.imshow(mask[0, 0, :, :].detach().cpu().numpy())
            # plt.savefig('F:\TEMP\\'+str(i)+'.png')
            # print(torch.max(u_raw_inter))

            loss = torch.sum(torch.abs(fine_disp + u_raw_inter.detach() - labels) * mask) #+ \
                   # 0.2 * torch.sum(torch.abs(raw_disp1 + cat_raw_uv_inter.detach() - labels) * mask) + \
                   # 0.1 * torch.sum(torch.abs(raw_disp2 + cat_raw_uv_inter.detach() - labels) * mask)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += torch.sum(torch.abs(fine_disp + u_raw_inter.detach() - labels) * mask).item() / torch.sum(mask).item()
            # print(torch.max((cat_raw_uv_inter.detach() - labels) * mask * (torch.abs(cat_raw_uv_inter.detach() - labels) < 3.0)).item())
            train_epe += torch.sum(torch.sum((fine_disp + u_raw_inter.detach() - labels) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            # train_epe += torch.sum(torch.sqrt(torch.sum((fine_disp - labels)**2*mask, dim=1))) / torch.sum(mask)
        mean_train_loss = mean_train_loss / i / 2
        train_epe = train_epe / i
        print(mean_train_loss, train_epe)

        if epoch % 2 == 0:
            # evaluator.show_seg_img()
            mean_loss = 0
            valid_epe = 0
            model.eval()
            for j, (images, labels, mask, fname) in enumerate(tqdm(valid_dataloader, desc="Validation", unit="iteration")):
                ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                # labels = labels
                mask = mask.to(device)
                mask[:, :, -20:, :] = False
                mask[:, :, :20, :] = False
                mask[:, :, :, -20:] = False
                mask[:, :, :, :20] = False
                # 前向传播
                pred_flow = model_GM(ref_imgs, def_imgs,
                                     attn_splits_list=[2],
                                     corr_radius_list=[-1],
                                     prop_radius_list=[-1])

                raw_disp = pred_flow['flow_preds'][-1][0, :, :, :].detach()
                raw_disp = my_raw_disp_blur(raw_disp)

                # 插值获得中间构型
                u_pos = grid_x + raw_disp[1]
                v_pos = grid_y + raw_disp[0]

                img_inter = my_interpolator.interpolation(u_pos, v_pos, def_imgs[0, 0, :, :], img_mode=False)

                # 精细像素位移计算
                imgs_array = torch.stack([ref_imgs[0, 0, :, :], img_inter], dim=0).unsqueeze(0).to(device)
                fine_disp = model(imgs_array.to(device))
                # 插值获得精细大像素位移
                [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
                    u_pos=grid_x + fine_disp.detach()[0][1],
                    v_pos=grid_y + fine_disp.detach()[0][0],
                    gray_array_list=[raw_disp.detach()[0], raw_disp.detach()[1]],
                    img_mode=False,
                    kernel='bspline'
                )
                u_raw_inter = torch.stack([u_raw_inter, v_raw_inter], dim=0).unsqueeze(0)
                mask_value = torch.abs(u_raw_inter.detach() - labels) < 5.0
                mask = mask * (mask_value[:, 0, :, :] * mask_value[:, 1, :, :] > 0).unsqueeze(1)

                # 插值获得精细大像素位移
                loss = torch.sum(torch.abs(fine_disp + u_raw_inter.detach() - labels.to(device)) * mask).item() / torch.sum(mask).item()
                # 反向传播和优化
                mean_loss += loss
                valid_epe += torch.sum(torch.sum((fine_disp.detach() + u_raw_inter - labels.to(device)) ** 2 * mask, dim=1).sqrt()).item() / torch.sum(mask).item()
            mean_loss = mean_loss / j / 2
            valid_epe = valid_epe / j
            # evaluator.show_seg_img()
            # 打印训练信息
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train epe Loss: {:.4f}, Valid Loss: {:.4f}, Valid epe Loss: {:.4f}'.
                  format(epoch + 1, num_epochs + steps, mean_train_loss, train_epe,
                         mean_loss, valid_epe))

            if mean_loss < max_loss:
                torch.save(model.state_dict(), savedir + '//SubpixelCorrNet_Best.pth')
                print('load: ', savedir + '//SubpixelCorrNet_Best.pth')
                max_loss = mean_loss
            torch.save(model.state_dict(), savedir + '//SubpixelCorrNet_Last.pth')


if __name__ == '__main__':
    dataset_dir = r'E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\\'
    # train_TransUNet_pret(dataset_dir=dataset_dir, scale=2)
    # train_SubpixelCorrNet_pret(dataset_dir=dataset_dir, load_init=False)
    train_SubpixelCorrNet_eval(dataset_dir=dataset_dir, load_init=False)