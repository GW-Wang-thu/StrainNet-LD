from networks.gmflow.gmflow import GMFlow
from networks.gmflow.loss import flow_loss_func
from utils.dataloader import GenerateDataloader
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.interpolation import interpolator
from torchsummary import summary


def train_GMFlowNet_pret(device='cuda', savedir='params', num_epochs=800, dataset_dir=r'E:\Data\DATASET\LargeDeformationDIC\202402\Dataset\\', load_init=False):
    batch_size = 2
    model = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   inchannel=3,
                   ).to(device)
    # print(summary(model, input_size=[(3, 128, 128), (3, 128, 128)]))
    train_dataloader = GenerateDataloader(dataset_dir, type='Train', batch_size=batch_size, scale=1, shift=True, pin_mem=False, training_size=768)
    valid_dataloader = GenerateDataloader(dataset_dir, type='Valid', batch_size=batch_size, scale=1, shift=True, pin_mem=False, training_size=768)
    # pre_trained_paramesters = torch.load(r'D:\Codes\.Git\Gmflow-main\pretrained\pretrained\\gmflow_chairs-1d776046.pth')
    # model.load_state_dict(pre_trained_paramesters['model'])
    if not load_init:
        if os.path.exists(savedir + '//GMFlowNet_Last.pth'):
            # with open(savedir + '//GMFlowNet_'+str(scale)+'_'+"_loss_rec.json", 'r') as f:
            #     loss_rec = json.load(f)
            # steps = 50 * ((loss_rec['train epoch'][-1] + 1) // 50)
            model.load_state_dict(torch.load(savedir + '//GMFlowNet_Last.pth'))
            print('load: ', savedir + '//GMFlowNet_Last.pth')
            max_loss = 10 #np.max(np.array(loss_rec['valid loss']))
        else:
            max_loss = 28
    else:
        pre_trained_paramesters = torch.load(r'D:\Codes\.Git\Gmflow-main\pretrained\pretrained\\gmflow_chairs-1d776046.pth')
        model.load_state_dict(pre_trained_paramesters['model'])
        max_loss = 28

    model.transformer.requires_grad_(False)
    model.feature_flow_attn.requires_grad_(False)

    steps = 0
    num_epochs -= steps
    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_rec = {'train epoch': [], 'valid epoch': [], 'train loss': [], 'valid loss': [], 'train epe': [], 'valid epe': []}
    # 训练模型
    for n in range(num_epochs):
        epoch = n + steps
        mean_train_loss = 0
        mean_train_loss_rec = 0
        train_epe = 0
        for i, (images, labels, mask, fname) in enumerate(tqdm(train_dataloader, desc="Training", unit="iteration")):
            # if n == 0:
            #     continue
            model.train()
            ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
            mask = mask.to(device)
            labels = labels.to(device)
            # 前向传播
            pred_flow = model(ref_imgs, def_imgs,
                              attn_splits_list=[2],
                              corr_radius_list=[-1],
                              prop_radius_list=[-1])
            loss, metrics = flow_loss_func(pred_flow['flow_preds'], labels.to(device), mask.to(device),
                                           gamma=0.9,
                                           max_flow=100,
                                           )

            # plt.subplot(2, 2, 1)
            # plt.imshow(pred_flow['flow_preds'][-1][0, 0, :, :].detach().cpu().numpy())
            # plt.colorbar()
            # plt.subplot(2, 2, 2)
            # plt.imshow(labels[0, 0, :, :].detach().cpu().numpy())
            # plt.colorbar()
            # plt.subplot(2, 2, 3)
            # plt.imshow(pred_flow['flow_preds'][-1][0, 1, :, :].detach().cpu().numpy())
            # plt.colorbar()
            # plt.subplot(2, 2, 4)
            # plt.imshow(labels[0, 1, :, :].detach().cpu().numpy())
            # plt.colorbar()
            # plt.title(fname[0].split('\\')[-1])
            # plt.show()

            # print('case loss: ', loss.item())
            train_epe += metrics['epe']

            # 反向传播和优化
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            mean_train_loss += loss.item()
            mean_train_loss_rec += torch.sum(torch.abs(pred_flow['flow_preds'][-1] - labels) * mask).item() / torch.sum(mask).item()
        mean_train_loss = mean_train_loss / i
        mean_train_loss_rec = mean_train_loss_rec / i / 2
        train_epe = train_epe / i
        loss_rec['train epoch'].append(epoch)
        loss_rec['train loss'].append(mean_train_loss)
        loss_rec['train epe'].append(train_epe)
        print(train_epe, mean_train_loss_rec)

        if epoch % 2 == 0:
            # evaluator.show_seg_img()
            model.eval()
            mean_loss = 0
            valid_epe = 0
            mean_loss_rec = 0.0
            for j, (images, labels, mask, fname) in enumerate(tqdm(valid_dataloader, desc="Validation", unit="iteration")):
                # 前向传播
                ref_imgs = images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                def_imgs = images[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1).cuda()
                # 前向传播
                pred_flow = model(ref_imgs, def_imgs,
                                  attn_splits_list=[2],
                                  corr_radius_list=[-1],
                                  prop_radius_list=[-1])
                loss, metrics = flow_loss_func(pred_flow['flow_preds'], labels.to(device), mask.to(device),
                                               gamma=0.9,
                                               max_flow=100,
                                               )
                mean_loss += loss.item()
                valid_epe += metrics['epe']
                mean_loss_rec += torch.sum(torch.abs(pred_flow['flow_preds'][-1].detach() - labels.to(device)) * mask.to(device)).item() / torch.sum(mask.to(device)).item()
            mean_loss = mean_loss / j
            valid_epe = valid_epe / j
            mean_loss_rec = mean_loss_rec / j / 2
            loss_rec['valid epoch'].append(epoch)
            if mean_loss > 0:
                loss_rec['valid loss'].append(mean_loss)
                loss_rec['valid epe'].append(valid_epe)
            # evaluator.show_seg_img()
            # 打印训练信息
            print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, train epe: {:.4f}, Valid epe: {:.4f}, train loss_rec: {:.4f}, Valid loss_rec: {:.4f}'.
                  format(epoch + 1, num_epochs + steps, mean_train_loss,
                         mean_loss,
                         train_epe,
                         valid_epe, mean_train_loss_rec, mean_loss_rec))

            if len(loss_rec['valid loss']) > 2:
                if mean_loss < max_loss:
                    torch.save(model.state_dict(), savedir + '//GMFlowNet_Best.pth')
                    print('save: ', savedir + '//GMFlowNet_Best.pth')
                    max_loss = mean_loss
        if epoch % 10 == 9:
            torch.save(model.state_dict(), savedir + '//GMFlowNet_Last.pth')

            with open(savedir + '//GMFlowNet_loss_rec.json', "w") as f:
                json.dump(loss_rec, f)



if __name__ == '__main__':
    dataset_dir = r'E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\\'
    train_GMFlowNet_pret(dataset_dir=dataset_dir, load_init=True)