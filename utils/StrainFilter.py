import cv2
import numpy as np
import torch
import sympy as sy
import time
import random
import matplotlib.pyplot as plt


class StrainFilter():
    def __init__(self, device='cpu', radius=7, skip=1):
        # a = 0.5
        # self.kernal_matrix = torch.from_numpy(np.array([[-0.5, 1.5, -1.5, 0.5],
        #                                                 [1.0, -2.5, 2.0, -0.5],
        #                                                 [-0.5, 0, 0.5, 0],
        #                                                 [0.0, 1.0, 0.0, 0.0]], dtype="float32")).to(self.device)
        # bicubic
        self.device = device
        self.radius = int(radius)
        self.xy_list = np.array([[x-radius, y-radius] for x in range(2 * radius+1) for y in range(2*radius+1) if ((x-radius)**2 + (y-radius)**2) <= radius**2])
        self.n = self.xy_list.shape[0] // skip
        self.skip = skip

    def fit_strain(self, u, v):
        imsize = u.shape
        padded_u = torch.zeros(size=(imsize[0] + 2 * self.radius, imsize[1] + 2 * self.radius)).long().to(self.device)
        padded_v = torch.zeros(size=(imsize[0] + 2 * self.radius, imsize[1] + 2 * self.radius)).long().to(self.device)
        padded_u[self.radius:-self.radius, self.radius:-self.radius] = u
        padded_v[self.radius:-self.radius, self.radius:-self.radius] = v
        u_pos, v_pos = torch.meshgrid(torch.arange(imsize[0]), torch.arange(imsize[1]))
        u_pos = u_pos.long().to(self.device) + self.radius
        v_pos = v_pos.long().to(self.device) + self.radius
        u_pos_vect = u_pos.flatten().to(self.device)
        v_pos_vect = v_pos.flatten().to(self.device)

        take_positions_x = []
        take_positions_y = []
        take_positions_mask = []
        take_u = []
        take_v = []
        max_x_pos = 0
        random_list = random.sample(self.xy_list.tolist(), self.n)
        for i in range(self.n):
            # i = int(i*self.skip)
            tmp_x_pos = u_pos + int(random_list[i][0])
            tmp_x_pos = tmp_x_pos.flatten().to(self.device).to(torch.long)
            tmp_y_pos = v_pos + int(random_list[i][1])
            tmp_y_pos = tmp_y_pos.flatten().to(self.device).to(torch.long)
            tmp_mask = (tmp_x_pos >= self.radius) * (tmp_x_pos < (self.radius+imsize[0])) * (tmp_y_pos >= self.radius) * (tmp_y_pos < (self.radius+imsize[1]))
            tmp_u_take = torch.take(padded_u.flatten(), tmp_x_pos * (imsize[1]+2*self.radius) + tmp_y_pos)
            tmp_v_take = torch.take(padded_v.flatten(), tmp_x_pos * (imsize[1]+2*self.radius) + tmp_y_pos)
            max_x_pos = max(torch.max(tmp_x_pos).item(), max_x_pos)
            take_positions_x.append(tmp_x_pos - u_pos_vect)
            take_positions_y.append(tmp_y_pos - v_pos_vect)
            take_positions_mask.append(tmp_mask)
            take_u.append(tmp_u_take)
            take_v.append(tmp_v_take)
        take_positions_mask = torch.stack(take_positions_mask, dim=0).to(torch.float32)
        take_positions_x = torch.stack(take_positions_x, dim=0).to(torch.float32) * take_positions_mask
        take_positions_y = torch.stack(take_positions_y, dim=0).to(torch.float32) * take_positions_mask
        take_u = torch.stack(take_u, dim=0) * take_positions_mask
        take_v = torch.stack(take_v, dim=0) * take_positions_mask
        take_u = (take_u - torch.sum(take_u, dim=0) / torch.sum(take_positions_mask, dim=0)) * take_positions_mask
        take_v = (take_v - torch.sum(take_v, dim=0) / torch.sum(take_positions_mask, dim=0)) * take_positions_mask
        # varia_u = torch.max(take_u, dim=0).values - torch.min(take_u, dim=0).values
        # varia_v = torch.max(take_v, dim=0).values - torch.min(take_v, dim=0).values
        # take_u = take_u * self.n / varia_u
        # take_v = take_u * self.n / varia_v

        # 组成3*3数组
        a = torch.sum(take_positions_x**2, dim=0)
        b = torch.sum(take_positions_x*take_positions_y, dim=0)
        c = torch.sum(take_positions_x, dim=0)
        d = torch.sum(take_positions_y**2, dim=0)
        e = torch.sum(take_positions_y, dim=0)
        f = torch.sum(take_positions_mask, dim=0)

        # 非齐次向量
        vec_u_1 = torch.sum(take_positions_x*take_u, dim=0)
        vec_u_2 = torch.sum(take_positions_y*take_u, dim=0)
        vec_u_3 = torch.sum(take_u, dim=0)

        vec_v_1 = torch.sum(take_positions_x*take_v, dim=0)
        vec_v_2 = torch.sum(take_positions_y*take_v, dim=0)
        vec_v_3 = torch.sum(take_v, dim=0)

        # 求逆求解矩阵

        inverse_mat_1_1 = (d*f - e**2)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)
        inverse_mat_1_2 = (-b*f + c*e)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)
        inverse_mat_1_3 = (b*e - c*d)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)
        inverse_mat_2_1 = (-b*f + c*e)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)
        inverse_mat_2_2 = (a*f - c**2)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)
        inverse_mat_2_3 = (-a*e + b*c)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)
        inverse_mat_3_1 = (b*e - c*d)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)
        inverse_mat_3_2 = (-a*e + b*c)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)
        inverse_mat_3_3 = (a*d - b**2)/(a*d*f - a*e**2 - b**2*f + 2*b*c*e - c**2*d)

        solved_u_x = inverse_mat_1_1 * vec_u_1 + inverse_mat_1_2 * vec_u_2 + inverse_mat_1_3 + vec_u_3
        solved_u_y = inverse_mat_2_1 * vec_u_1 + inverse_mat_2_2 * vec_u_2 + inverse_mat_2_3 + vec_u_3
        # solved_u_0 = inverse_mat_3_1 * vec_u_1 + inverse_mat_3_2 * vec_u_2 + inverse_mat_3_3 + vec_u_3

        solved_v_x = inverse_mat_1_1 * vec_v_1 + inverse_mat_1_2 * vec_v_2 + inverse_mat_1_3 + vec_v_3
        solved_v_y = inverse_mat_2_1 * vec_v_1 + inverse_mat_2_2 * vec_v_2 + inverse_mat_2_3 + vec_v_3
        # solved_v_0 = inverse_mat_3_1 * vec_v_1 + inverse_mat_3_2 * vec_v_2 + inverse_mat_3_3 + vec_v_3

        # solved_u_x = solved_u_x * varia_u / self.n
        # solved_u_y = solved_u_y * varia_u / self.n
        # solved_v_x = solved_v_x * varia_v / self.n
        # solved_v_y = solved_v_y * varia_v / self.n

        solved_u_x = solved_u_x.unflatten(dim=0, sizes=imsize)
        solved_u_y = solved_u_y.unflatten(dim=0, sizes=imsize)
        solved_v_x = solved_v_x.unflatten(dim=0, sizes=imsize)
        solved_v_y = solved_v_y.unflatten(dim=0, sizes=imsize)

        return solved_u_y, solved_u_x, solved_v_y, solved_v_x


def solve_inverse():
    from sympy import symbols, Matrix
    a, b, c, d, e, f,  = symbols('a b c d e f')
    A = Matrix([[a, b, c],
                [b, d, e],
                [c, e, f]])
    A_inv = A.inv()
    print(A_inv)


if __name__ == '__main__':
    # solve_inverse()
    my_strain_differ = StrainFilter(device='cuda', radius=18, skip=2)
    u_pos, v_pos = torch.meshgrid(torch.arange(0, 256), torch.arange(0, 125))
    u_pos = u_pos * v_pos * 0.01 + torch.randn(size=u_pos.shape) * 0.1
    v_pos = u_pos * v_pos * 0.01 + torch.randn(size=u_pos.shape) * 0.1

    solved_u_x, solved_u_y, solved_v_x, solved_v_y = my_strain_differ.fit_strain(u_pos.cuda(), v_pos.cuda())
    # plt.imshow(u_pos.cpu().numpy())
    # plt.colorbar()
    # plt.show()
    # plt.imshow(v_pos.cpu().numpy())
    # plt.show()
    plt.imshow(solved_u_x.cpu().numpy()[20:-20, 20:-20])
    plt.colorbar()
    plt.show()
    plt.imshow((u_pos[1:, :]-u_pos[:-1, :]).cpu().numpy()[20:-20, 20:-20])
    plt.colorbar()
    plt.show()
    plt.imshow(solved_u_y.cpu().numpy()[20:-20, 20:-20])
    plt.colorbar()
    plt.show()
    plt.imshow((v_pos[:, 1:]-v_pos[:, :-1]).cpu().numpy()[20:-20, 20:-20])
    plt.colorbar()
    plt.show()
    plt.imshow(solved_v_y.cpu().numpy()[20:-20, 20:-20])
    plt.colorbar()
    plt.show()

