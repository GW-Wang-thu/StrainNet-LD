import torch.nn as nn
import torch
import numpy as np
from utils.StrainFilter import StrainFilter


class DifferNet(nn.Module):
    def __init__(self, device='cuda'):
        super(DifferNet, self).__init__()
        self.differ_kernel_x = (1 / 5) * torch.from_numpy(np.array([[-0.0625, -0.125, 0, 0.125, 0.0625],
                                                                    [-0.125, -0.25, 0, 0.25, 0.125],
                                                                    [-0.25, -0.5, 0, 0.5, 0.25],
                                                                    [-0.125, -0.25, 0, 0.25, 0.125],
                                                                    [-0.0625, -0.125, 0, 0.125, 0.0625]],
                                                                   dtype='float32')).unsqueeze(0).unsqueeze(0).to(device)
        self.differ_kernel_y = (1 / 5) * torch.from_numpy(np.array([[-0.0625, -0.125, 0, 0.125, 0.0625],
                                                                    [-0.125, -0.25, 0, 0.25, 0.125],
                                                                    [-0.25, -0.5, 0, 0.5, 0.25],
                                                                    [-0.125, -0.25, 0, 0.25, 0.125],
                                                                    [-0.0625, -0.125, 0, 0.125, 0.0625]],
                                                                   dtype='float32').T).unsqueeze(0).unsqueeze(0).to(device)
        self.blur_layers = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                         nn.ReLU().to(device),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                         nn.ReLU().to(device),
                                         nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device))

        self.conv_layer = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device)
        self.differ_layer_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False).to(device)
        self.differ_layer_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False).to(device)
        self.differ_layer_x.requires_grad_(False)
        self.differ_layer_y.requires_grad_(False)
        self.differ_layer_x.weight.data = self.differ_kernel_x
        self.differ_layer_y.weight.data = self.differ_kernel_y

    def forward(self, displacements, with_net=False):

        disp_x = displacements[:, 0, :, :].unsqueeze(1)
        disp_y = displacements[:, 1, :, :].unsqueeze(1)
        if with_net:
            disp_x = disp_x + self.blur_layers(disp_x)
            disp_y = disp_y + self.blur_layers(disp_y)
        exx = self.differ_layer_x(disp_x)
        exy = self.differ_layer_y(disp_x)
        eyx = self.differ_layer_x(disp_y)
        eyy = self.differ_layer_y(disp_y)
        strain_raw = torch.cat([exx, exy, eyx, eyy], dim=1)
        if with_net:
            pred_strain = self.conv_layer(strain_raw)
        else:
            pred_strain = strain_raw
        return pred_strain


class DifferNet_FIT(nn.Module):
    def __init__(self, device='cuda', radius=11, skip=2):
        super(DifferNet_FIT, self).__init__()
        self.strain_fit = StrainFilter(device=device, radius=radius, skip=skip)
        self.blur_layers = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
                                         nn.ReLU().to(device),
                                         nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)).to(device),
                                         nn.ReLU().to(device),
                                         nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
                                         nn.ReLU().to(device),
                                         nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                         nn.ReLU().to(device),
                                         nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)).to(device),
                                         nn.ReLU().to(device),
                                         nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device))

        self.conv_layer = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device)

    def forward(self, displacements, with_net=False):
        b, _, _, _ = displacements.shape
        disp_x = displacements[:, 0, :, :].unsqueeze(1)
        disp_y = displacements[:, 1, :, :].unsqueeze(1)
        if with_net:
            strain_residue = self.blur_layers(displacements)
        exx_list = []
        exy_list = []
        eyx_list = []
        eyy_list = []
        for i in range(b):
            exx, exy, eyx, eyy = self.strain_fit.fit_strain(disp_x[0, 0, :, :] * 100.0, disp_y[0, 0, :, :] * 100.0)
            exx_list.append(exx / 100.0)
            exy_list.append(exy / 100.0)
            eyx_list.append(eyx / 100.0)
            eyy_list.append(eyy / 100.0)
        exx = torch.stack(exx_list, dim=0)
        exy = torch.stack(exy_list, dim=0)
        eyx = torch.stack(eyx_list, dim=0)
        eyy = torch.stack(eyy_list, dim=0)
        strain_raw = torch.cat([exx.unsqueeze(0), exy.unsqueeze(0), eyx.unsqueeze(0), eyy.unsqueeze(0)], dim=1)
        if with_net:
            strain_raw = torch.cat([strain_residue, strain_raw], dim=1)
            pred_strain = self.conv_layer(strain_raw)
        else:
            pred_strain = strain_raw
        return pred_strain


class DifferNet_FIT_Inter(nn.Module):
    def __init__(self, device='cuda', radius=11, skip=2):
        super(DifferNet_FIT_Inter, self).__init__()
        self.strain_fit = StrainFilter(device=device, radius=radius, skip=skip)
        self.block_1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)).to(device),
                                     nn.ReLU().to(device))
        self.block_2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device))
        self.block_3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device))
        self.block_4 = nn.Sequential(nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)).to(device),
                                     nn.Tanh().to(device),
                                     nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device))
        # self.disp_block = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
        #                                 nn.Tanh().to(device),
        #                                 nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
        #                                 nn.Tanh().to(device),
        #                                 nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
        #                                 nn.Tanh().to(device),
        #                                 nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device))

    def forward(self, displacements, with_net=False):
        if not with_net:
            b, _, _, _ = displacements.shape
            disp_x = displacements[:, 0, :, :].unsqueeze(1)
            disp_y = displacements[:, 1, :, :].unsqueeze(1)
            exx_list = []
            exy_list = []
            eyx_list = []
            eyy_list = []
            for i in range(b):
                exx, exy, eyx, eyy = self.strain_fit.fit_strain(disp_x[0, 0, :, :] * 100.0, disp_y[0, 0, :, :] * 100.0)
                exx_list.append(exx / 100.0)
                exy_list.append(exy / 100.0)
                eyx_list.append(eyx / 100.0)
                eyy_list.append(eyy / 100.0)
            exx = torch.stack(exx_list, dim=0)
            exy = torch.stack(exy_list, dim=0)
            eyx = torch.stack(eyx_list, dim=0)
            eyy = torch.stack(eyy_list, dim=0)
            pred_strain = torch.cat([exx.unsqueeze(0), exy.unsqueeze(0), eyx.unsqueeze(0), eyy.unsqueeze(0)], dim=1)
        else:
            strain_residue = self.block_1(displacements)
            strain_residue = self.block_2(strain_residue) + strain_residue
            strain_residue = self.block_3(strain_residue) + strain_residue
            pred_strain = self.block_4(strain_residue) / 100.0
            # pred_strain = pred_strain + self.disp_block(fine_disp) / 100.0
        return pred_strain


class DifferNet_FIT_Inter_L(nn.Module):
    def __init__(self, device='cuda', radius=11, skip=2):
        super(DifferNet_FIT_Inter_L, self).__init__()
        self.strain_fit = StrainFilter(device=device, radius=radius, skip=skip)
        self.block_1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)).to(device),
                                     nn.ReLU().to(device))
        self.block_2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device))
        self.block_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device))
        self.block_4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device))
        self.block_5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
                                     nn.ReLU().to(device))
        self.block_6 = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)).to(device),
                                     nn.Tanh().to(device),
                                     nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device))
        # self.disp_block = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
        #                                 nn.Tanh().to(device),
        #                                 nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)).to(device),
        #                                 nn.Tanh().to(device),
        #                                 nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device),
        #                                 nn.Tanh().to(device),
        #                                 nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device))

    def forward(self, displacements, with_net=False):
        if not with_net:
            b, _, _, _ = displacements.shape
            disp_x = displacements[:, 0, :, :].unsqueeze(1)
            disp_y = displacements[:, 1, :, :].unsqueeze(1)
            exx_list = []
            exy_list = []
            eyx_list = []
            eyy_list = []
            for i in range(b):
                exx, exy, eyx, eyy = self.strain_fit.fit_strain(disp_x[0, 0, :, :] * 100.0, disp_y[0, 0, :, :] * 100.0)
                exx_list.append(exx / 100.0)
                exy_list.append(exy / 100.0)
                eyx_list.append(eyx / 100.0)
                eyy_list.append(eyy / 100.0)
            exx = torch.stack(exx_list, dim=0)
            exy = torch.stack(exy_list, dim=0)
            eyx = torch.stack(eyx_list, dim=0)
            eyy = torch.stack(eyy_list, dim=0)
            pred_strain = torch.cat([exx.unsqueeze(0), exy.unsqueeze(0), eyx.unsqueeze(0), eyy.unsqueeze(0)], dim=1)
        else:
            strain_residue = self.block_1(displacements)
            strain_residue = self.block_2(strain_residue)
            strain_residue = self.block_3(strain_residue) + strain_residue
            strain_residue = self.block_4(strain_residue) + strain_residue
            strain_residue = self.block_5(strain_residue) + strain_residue
            pred_strain = self.block_6(strain_residue) / 100.0
            # pred_strain = pred_strain + self.disp_block(fine_disp) / 100.0
        return pred_strain
