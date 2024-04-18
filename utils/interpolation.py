import cv2
import numpy as np
import torch
import sympy as sy
import time
import matplotlib.pyplot as plt


class interpolator():
    def __init__(self, device='cpu'):
        # a = 0.5
        # self.kernal_matrix = torch.from_numpy(np.array([[-0.5, 1.5, -1.5, 0.5],
        #                                                 [1.0, -2.5, 2.0, -0.5],
        #                                                 [-0.5, 0, 0.5, 0],
        #                                                 [0.0, 1.0, 0.0, 0.0]], dtype="float32")).to(self.device)
        # bicubic
        self.device = device
        self.kernal_matrix_bicubic = torch.from_numpy(np.array([[-0.75, 1.25, -1.25, 0.75],
                                                        [1.5, -2.25, 1.5, -0.75],
                                                        [-0.75, 0, 0.75, 0],
                                                        [0.0, 1.0, 0.0, 0.0]], dtype="float32")).to(self.device)
        # b_spline
        self.kernal_matrix_spline = torch.from_numpy(np.array([[-0.166666666666667, 0.5, -0.5, 0.166666666666667],
                                                        [0.5, -1.0, 0.5, 0],
                                                        [-0.5, 0, 0.5, 0],
                                                        [0.166666666666667, 0.666666666666667, 0.166666666666667, 0]], dtype="float32")).to(self.device)

    def interpolation(self, u_pos, v_pos, gray_array, img_mode=True, kernel="bicubic"):
        imsize = gray_array.shape
        if kernel == 'bicubic':
            kernel_matrix = self.kernal_matrix_bicubic
        else:
            kernel_matrix = self.kernal_matrix_spline

        target_imsize = u_pos.shape

        u_pos = u_pos.flatten().to(self.device)
        v_pos = v_pos.flatten().to(self.device)
        gray_array = gray_array.flatten().to(self.device)

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0]-1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1]-1)

        pos_x_m1 = pos_x_0 - 1
        mask_x_m1 = (pos_x_m1 >= 0) * (pos_x_m1 <= imsize[0]-1)

        pos_y_m1 = pos_y_0 - 1
        mask_y_m1 = (pos_y_m1 >= 0) * (pos_y_m1 <= imsize[1]-1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0]-1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1]-1)

        pos_x_2 = pos_x_0 + 2
        mask_x_2 = (pos_x_2 >= 0) * (pos_x_2 <= imsize[0]-1)

        pos_y_2 = pos_y_0 + 2
        mask_y_2 = (pos_y_2 >= 0) * (pos_y_2 <= imsize[1]-1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_m1 = (pos_x_m1 * mask_x_m1)
        pos_y_m1 = (pos_y_m1 * mask_y_m1)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)
        pos_x_2 = (pos_x_2 * mask_x_2)
        pos_y_2 = (pos_y_2 * mask_y_2)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)

        g_m1_m1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_m1) * mask_x_m1 * mask_y_m1
        g_m1_0 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_0) * mask_x_m1 * mask_y_0
        g_m1_1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_1) * mask_x_m1 * mask_y_1
        g_m1_2 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_2) * mask_x_m1 * mask_y_2
        g_0_m1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_m1) * mask_x_0 * mask_y_m1
        g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
        g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
        g_0_2 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_2) * mask_x_0 * mask_y_2
        g_1_m1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_m1) * mask_x_1 * mask_y_m1
        g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
        g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1
        g_1_2 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_2) * mask_x_1 * mask_y_2
        g_2_m1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_m1) * mask_x_2 * mask_y_m1
        g_2_0 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_0) * mask_x_2 * mask_y_0
        g_2_1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_1) * mask_x_2 * mask_y_1
        g_2_2 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_2) * mask_x_2 * mask_y_2

        t_x_vect = torch.column_stack([tx_vect**3, tx_vect**2, tx_vect, torch.ones_like(tx_vect).to(self.device)])
        t_y_vect = torch.column_stack([ty_vect**3, ty_vect**2, ty_vect, torch.ones_like(ty_vect).to(self.device)])
        b = torch.matmul(t_x_vect, kernel_matrix).unsqueeze(2)
        a = torch.matmul(t_y_vect, kernel_matrix).unsqueeze(1)

        ratios = torch.reshape(torch.bmm(b, a), (-1, 16))
        cated_grays = torch.column_stack([g_m1_m1, g_m1_0, g_m1_1, g_m1_2,
                                          g_0_m1, g_0_0, g_0_1, g_0_2,
                                          g_1_m1, g_1_0, g_1_1, g_1_2,
                                          g_2_m1, g_2_0, g_2_1, g_2_2])
        g_inter = torch.sum(ratios * cated_grays, dim=-1)
        img_float = g_inter.unflatten(dim=0, sizes=target_imsize)
        if img_mode:
            img = (img_float - 255) * (img_float <= 255) + 255
            img = img * (img >= 0)
            return img
        else:
            return img_float

    def interpolation_list(self, u_pos, v_pos, gray_array_list, img_mode=True, kernel="bicubic"):
        imsize = gray_array_list[0].shape
        if kernel == 'bicubic':
            kernel_matrix = self.kernal_matrix_bicubic
        else:
            kernel_matrix = self.kernal_matrix_spline
        target_imsize = u_pos.shape
        u_pos = u_pos.flatten().to(self.device)
        v_pos = v_pos.flatten().to(self.device)

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0]-1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1]-1)

        pos_x_m1 = pos_x_0 - 1
        mask_x_m1 = (pos_x_m1 >= 0) * (pos_x_m1 <= imsize[0]-1)

        pos_y_m1 = pos_y_0 - 1
        mask_y_m1 = (pos_y_m1 >= 0) * (pos_y_m1 <= imsize[1]-1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0]-1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1]-1)

        pos_x_2 = pos_x_0 + 2
        mask_x_2 = (pos_x_2 >= 0) * (pos_x_2 <= imsize[0]-1)

        pos_y_2 = pos_y_0 + 2
        mask_y_2 = (pos_y_2 >= 0) * (pos_y_2 <= imsize[1]-1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_m1 = (pos_x_m1 * mask_x_m1)
        pos_y_m1 = (pos_y_m1 * mask_y_m1)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)
        pos_x_2 = (pos_x_2 * mask_x_2)
        pos_y_2 = (pos_y_2 * mask_y_2)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)

        t_x_vect = torch.column_stack([tx_vect ** 3, tx_vect ** 2, tx_vect, torch.ones_like(tx_vect).to(self.device)])
        t_y_vect = torch.column_stack([ty_vect ** 3, ty_vect ** 2, ty_vect, torch.ones_like(ty_vect).to(self.device)])
        b = torch.matmul(t_x_vect, kernel_matrix).unsqueeze(2)
        a = torch.matmul(t_y_vect, kernel_matrix).unsqueeze(1)

        ratios = torch.reshape(torch.bmm(b, a), (-1, 16))

        results = []
        for i in range(len(gray_array_list)):
            gray_array = gray_array_list[i].flatten().to(self.device)
            g_m1_m1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_m1) * mask_x_m1 * mask_y_m1
            g_m1_0 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_0) * mask_x_m1 * mask_y_0
            g_m1_1 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_1) * mask_x_m1 * mask_y_1
            g_m1_2 = torch.take(gray_array, pos_x_m1 * imsize[1] + pos_y_2) * mask_x_m1 * mask_y_2
            g_0_m1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_m1) * mask_x_0 * mask_y_m1
            g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
            g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
            g_0_2 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_2) * mask_x_0 * mask_y_2
            g_1_m1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_m1) * mask_x_1 * mask_y_m1
            g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
            g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1
            g_1_2 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_2) * mask_x_1 * mask_y_2
            g_2_m1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_m1) * mask_x_2 * mask_y_m1
            g_2_0 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_0) * mask_x_2 * mask_y_0
            g_2_1 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_1) * mask_x_2 * mask_y_1
            g_2_2 = torch.take(gray_array, pos_x_2 * imsize[1] + pos_y_2) * mask_x_2 * mask_y_2
            cated_grays = torch.column_stack([g_m1_m1, g_m1_0, g_m1_1, g_m1_2,
                                              g_0_m1, g_0_0, g_0_1, g_0_2,
                                              g_1_m1, g_1_0, g_1_1, g_1_2,
                                              g_2_m1, g_2_0, g_2_1, g_2_2])
            g_inter = torch.sum(ratios * cated_grays, dim=-1)
            img_float = g_inter.unflatten(dim=0, sizes=target_imsize)

            if img_mode:
                img = (img_float - 255) * (img_float <= 255) + 255
                img = img * (img >= 0)
                results.append(img)
            else:
                results.append(img_float)
        return results

    def interpolation_bilinear(self, u_pos, v_pos, gray_array, img_mode=True):

        imsize = gray_array.shape
        target_imsize = u_pos.shape

        u_pos = u_pos.flatten().to(self.device)
        v_pos = v_pos.flatten().to(self.device)
        gray_array = gray_array.flatten().to(self.device)

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0] - 1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1] - 1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0] - 1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1] - 1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)

        g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
        g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
        g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
        g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1

        a_00 = g_0_0
        a_10 = - g_0_0 + g_1_0
        a_01 = - g_0_0 + g_0_1
        a_11 = g_0_0 - g_0_1 - g_1_0 + g_1_1

        g_inter = a_00 + a_10 * tx_vect + a_01 * ty_vect + a_11 * tx_vect * ty_vect
        img_float = g_inter.unflatten(dim=0, sizes=target_imsize)
        if img_mode:
            img = (img_float - 255) * (img_float <= 255) + 255
            img = img * (img >= 0)
            return img
        else:
            return img_float

    def interpolation_bilinear_list(self, u_pos, v_pos, gray_array_list, img_mode=True):

        imsize = gray_array_list[0].shape
        target_imsize = u_pos.shape

        u_pos = u_pos.flatten().to(self.device)
        v_pos = v_pos.flatten().to(self.device)

        pos_x_0 = torch.floor(u_pos).to(torch.long)
        mask_x_0 = (pos_x_0 >= 0) * (pos_x_0 <= imsize[0] - 1)

        pos_y_0 = torch.floor(v_pos).to(torch.long)
        mask_y_0 = (pos_y_0 >= 0) * (pos_y_0 <= imsize[1] - 1)

        pos_x_1 = pos_x_0 + 1
        mask_x_1 = (pos_x_1 >= 0) * (pos_x_1 <= imsize[0] - 1)

        pos_y_1 = pos_y_0 + 1
        mask_y_1 = (pos_y_1 >= 0) * (pos_y_1 <= imsize[1] - 1)

        pos_x_0 = (pos_x_0 * mask_x_0)
        pos_y_0 = (pos_y_0 * mask_y_0)
        pos_x_1 = (pos_x_1 * mask_x_1)
        pos_y_1 = (pos_y_1 * mask_y_1)

        tx_vect = u_pos - torch.floor(u_pos)
        ty_vect = v_pos - torch.floor(v_pos)


        results = []
        for i in range(len(gray_array_list)):
            gray_array = gray_array_list[i].flatten().to(self.device)
            g_0_0 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_0) * mask_x_0 * mask_y_0
            g_0_1 = torch.take(gray_array, pos_x_0 * imsize[1] + pos_y_1) * mask_x_0 * mask_y_1
            g_1_0 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_0) * mask_x_1 * mask_y_0
            g_1_1 = torch.take(gray_array, pos_x_1 * imsize[1] + pos_y_1) * mask_x_1 * mask_y_1

            a_00 = g_0_0
            a_10 = - g_0_0 + g_1_0
            a_01 = - g_0_0 + g_0_1
            a_11 = g_0_0 - g_0_1 - g_1_0 + g_1_1

            g_inter = a_00 + a_10 * tx_vect + a_01 * ty_vect + a_11 * tx_vect * ty_vect
            img_float = g_inter.unflatten(dim=0, sizes=target_imsize)

            if img_mode:
                img = (img_float - 255) * (img_float <= 255) + 255
                img = img * (img >= 0)
                results.append(img)
            else:
                results.append(img_float)
        return results
