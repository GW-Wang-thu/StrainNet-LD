import os.path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from eval import StrainNet_LD, StrainNet_L3D
from scipy.io import savemat, loadmat

ncorr_results_path = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr'
label_path = r'E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\Valid'


def load_ncorr_mat_result_seq(fname):
    displacement_u = []
    displacement_v = []
    mask_dice = []

    fname = os.path.join(ncorr_results_path, fname)
    tmp_ncorr_mat = loadmat(fname)

    for i in range((len(tmp_ncorr_mat.keys()) - 3) // 2):
        tmp_displacement_u = tmp_ncorr_mat['u'+str(i+1)].astype('float32')
        tmp_displacement_v = tmp_ncorr_mat['v'+str(i+1)].astype('float32')
        tmp_mask_dice = (tmp_displacement_u != 0) * (tmp_displacement_v != 0)
        displacement_u.append(tmp_displacement_u)
        displacement_v.append(tmp_displacement_v)
        mask_dice.append(tmp_mask_dice)
    return displacement_u, displacement_v, mask_dice


def ncorr_3D(case_id):
    dir = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\StereoEXP\DICe'
    ncorr_mat = os.path.join(dir, 'CASE_#'+str(case_id)+'_uv.mat')
    u_list, v_list, mask_list = load_ncorr_mat_result_seq(ncorr_mat)
    my_3d_recon = StrainNet_L3D(params_dir='../params/', device='cuda', raw_kernel_size=55, num_iter=iter, blockpos=[1000-256, 1000+1024+256, 1500-256, 1500+1024+256])
    uvw, surf_0, surf_1, flow, disparity_0 = my_3d_recon.calc_3DFlow(disparity_0=u_list[0], flow_x=u_list[1], flow_y=v_list[1], fd_x=u_list[2], debug=True)
    return uvw, surf_0, surf_1, mask_list[0], flow, disparity_0


def evaluate_net_case(case_id):
    my_correlator = StrainNet_L3D(params_dir='../params/', device='cuda', raw_kernel_size=55, num_iter=2, blockpos=[1000-256, 1000+1024+256, 1500-256, 1500+1024+256])
    path = r'E:\Data\Experiments\LargeDeformation\20240331\Sample'+str(case_id)+'\Selected'
    # img_l0 = cv2.resize(cv2.imread(os.path.join(path, '1_0.bmp'), cv2.IMREAD_GRAYSCALE), dsize=(768, 768), interpolation=cv2.INTER_CUBIC)
    # img_l1 = cv2.resize(cv2.imread(os.path.join(path, '1_2.bmp'), cv2.IMREAD_GRAYSCALE), dsize=(768, 768), interpolation=cv2.INTER_CUBIC)
    # img_r0 = cv2.resize(cv2.imread(os.path.join(path, '1_1.bmp'), cv2.IMREAD_GRAYSCALE), dsize=(768, 768), interpolation=cv2.INTER_CUBIC)
    # img_r1 = cv2.resize(cv2.imread(os.path.join(path, '1_3.bmp'), cv2.IMREAD_GRAYSCALE), dsize=(768, 768), interpolation=cv2.INTER_CUBIC)

    img_l0 = cv2.imread(os.path.join(path, '1_0.bmp'), cv2.IMREAD_GRAYSCALE)
    img_l1 = cv2.imread(os.path.join(path, '1_2.bmp'), cv2.IMREAD_GRAYSCALE)
    img_r0 = cv2.imread(os.path.join(path, '1_1.bmp'), cv2.IMREAD_GRAYSCALE)
    img_r1 = cv2.imread(os.path.join(path, '1_3.bmp'), cv2.IMREAD_GRAYSCALE)
    imsize = img_l0.shape

    mask = np.zeros_like(img_l0).astype('float32')
    mask[100:-100, 100:-100] = 1

    uvw, surf_0, surf_1, flow, disparity_0 = my_correlator.calc_3DFlow(L0=img_l0, L1=img_l1, R0=img_r0, R1=img_r1, mask=mask, imsize=imsize, debug=True)

    return uvw, surf_0, surf_1, mask, flow, disparity_0

    # plt.figure(figsize=(8, 6))  # 设置画布大小
    # ax = plt.axes(projection='3d')  # 设置三维轴
    # ax.scatter3D(surf_0[0][105:-105, 105:-105].flatten(), surf_0[1][105:-105, 105:-105].flatten(),
    #              surf_0[2][105:-105, 105:-105].flatten(), c=surf_0[2][105:-105, 105:-105].flatten())  # 三个数组对应三个维度（三个数组中的数一一对应）
    # # 显示图形
    # plt.show()

    # plt.imshow(surf_0[0]*mask)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(surf_0[1]*mask)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(surf_0[2]*mask)
    # plt.colorbar()
    # plt.show()

def Transform(points, theta_x, theta_y, Ty, T):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 假设 points 是一个 N×3 的矩阵
    # 例如: points = np.random.rand(100, 3)  # 随机生成 100 个三维点

    # 定义新的坐标系参数 (x1, y1, z1)
    # 这里我们使用一个简单的旋转和平移作为示例
    tx, ty, tz = 55, Ty, T  # 平移向量

    # 创建绕 x 轴的旋转矩阵 Rx
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    # 创建绕 y 轴的旋转矩阵 Ry
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    # 合并两个旋转矩阵得到总的旋转矩阵 R
    R0 = np.dot(Ry, Rx)
    R = np.zeros((4, 4))
    R[-1, -1] = 1
    R[:-1, :-1] = R0

    # 创建平移矩阵 T
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

    # 合并旋转矩阵和平移矩阵得到齐次变换矩阵 M
    M = np.dot(T, R)

    # 扩展 points 为 N×4 的矩阵，添加一列值为 1
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 应用齐次变换
    points_transformed = np.dot(points_homogeneous, M.T)  # 使用 M 的转置

    # 将变换后的齐次坐标转换回笛卡尔坐标
    points_transformed = points_transformed[:, :3] / points_transformed[:, 3].reshape(-1, 1)

    # 绘制原始点云和变换后的点云
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(str(round(theta_x, 4))+'_'+str(round(theta_y, 4)))

    # 绘制原始点云
    # ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2], c=points[::10, 2], label='Original Points')

    # 绘制变换后的点云
    ax.scatter(points_transformed[::10, 0], points_transformed[::10, 1], points_transformed[::10, 2], c=points_transformed[::10, 2],
               label='Transformed Points')

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 添加图例
    ax.legend()

    # 显示图形
    plt.show()
    return points_transformed[:, 0], points_transformed[:, 1], points_transformed[:, 2]


def evaluate_net_case_composite():
    path = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\StereoEXP\Composite'
    bd = 30
    id_list = [153005354, 157500715, 162505162]
    box_list = [[700-bd, 2440+bd, 500-bd, 2240+bd], [660-bd, 2400+bd, 630-bd, 2370+bd], [640-bd, 2380+bd, 600-bd, 2340+bd]]
    theta_x_l = [-0.075, -0.0553, -0.055]
    theta_y_l = [-0.0884, -0.0916, -0.0875]
    T_l = [-450-13, -450-12.5, -450-6]
    Ty_l = [-40, -40+10, -40+10]
    # id_list = [638903962, 643894650, 648393600]
    # box_list = [[700+30-bd, 2440+30+bd, 500+40-bd, 2240+40+bd], [660+50-bd, 2400+50+bd, 630-80-bd, 2370-80+bd], [640+120-bd, 2380+120+bd, 600-20-bd, 2340-20+bd]]
    # theta_x_l = [-0.055, -0.0897, -0.0628]
    # theta_y_l = [-0.0866, -0.0887, -0.0875]
    # T_l = [-450-13+4.5, -450-12.5+3.5, -450-6-5.4]
    # Ty_l = [-40+5, -40-10, -40+2]
    results = {}
    for i in range(len(id_list)):
        box = box_list[i]
        id = id_list[i]
        my_correlator = StrainNet_L3D(params_dir='../params/', device='cuda', raw_kernel_size=55, num_iter=2, blockpos=box)
        img_l0 = cv2.imread(os.path.join(path, 'IMG_'+str(id)+'_l.bmp'), cv2.IMREAD_GRAYSCALE)[box[0]:box[1], box[2]:box[3]]
        img_r0 = cv2.imread(os.path.join(path, 'IMG_'+str(id)+'_r.bmp'), cv2.IMREAD_GRAYSCALE)[box[0]:box[1], box[2]:box[3]]
        surf = my_correlator.calc_3DRecon(L0=img_l0, R0=img_r0, mask=None)
        x, y, z = Transform(np.array(surf).reshape(3, -1).T, theta_x=theta_x_l[i], theta_y=theta_y_l[i], T=T_l[i], Ty=Ty_l[i])

        results.update({'x_'+str(id): x.reshape(surf[0].shape[0], surf[0].shape[1]), 'y_'+str(id): y.reshape(surf[0].shape[0], surf[0].shape[1]), 'z_'+str(id): z.reshape(surf[0].shape[0], surf[0].shape[1])})
        # cv2.imshow('wind', img_l0)
        # cv2.waitKey(1000)
        # for theta_x in range(4):
        #     for theta_y in range(4):
        #         Transform(np.array(surf)[:, 50:-50, 50:-50].reshape(3, -1).T, theta_x=(theta_x-1.5) * np.pi / 1000+theta_x_l[i], theta_y=(theta_y-1.5) * np.pi / 1000+theta_y_l[i], T=T_l[i])

    savemat(os.path.join(path, 'results_trans_free.mat'), results)

    # plt.figure(figsize=(8, 6))  # 设置画布大小
    # ax = plt.axes(projection='3d')  # 设置三维轴
    # ax.scatter3D(surf_0[0][105:-105, 105:-105].flatten(), surf_0[1][105:-105, 105:-105].flatten(),
    #              surf_0[2][105:-105, 105:-105].flatten(), c=surf_0[2][105:-105, 105:-105].flatten())  # 三个数组对应三个维度（三个数组中的数一一对应）
    # # 显示图形
    # plt.show()

    # plt.imshow(surf_0[0]*mask)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(surf_0[1]*mask)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(surf_0[2]*mask)
    # plt.colorbar()
    # plt.show()

def evaluate_net_case_composite_single():
    path = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\StereoEXP\Composite'
    bd = 30
    id_list = [12523569]
    box_list = [[380-bd, 1960+bd, 1020-bd, 2600+bd]]
    # id_list = [638903962, 643894650, 648393600]
    # box_list = [[700+30-bd, 2440+30+bd, 500+40-bd, 2240+40+bd], [660+50-bd, 2400+50+bd, 630-80-bd, 2370-80+bd], [640+120-bd, 2380+120+bd, 600-20-bd, 2340-20+bd]]

    for i in range(len(id_list)):
        results = {}
        box = box_list[i]
        id = id_list[i]
        my_correlator = StrainNet_L3D(params_dir='../params/', device='cuda', raw_kernel_size=55, num_iter=2, blockpos=box)
        img_l0 = cv2.imread(os.path.join(path, 'IMG_'+str(id)+'_l.bmp'), cv2.IMREAD_GRAYSCALE)[box[0]:box[1], box[2]:box[3]]
        img_r0 = cv2.imread(os.path.join(path, 'IMG_'+str(id)+'_r.bmp'), cv2.IMREAD_GRAYSCALE)[box[0]:box[1], box[2]:box[3]]
        surf = my_correlator.calc_3DRecon(L0=img_l0, R0=img_r0, mask=None)
        results.update({'x': surf[0], 'y': surf[1], 'z': surf[2]})
        cv2.imshow('wind', img_l0)
        cv2.waitKey(1000)
        savemat(os.path.join(path, str(id)+'_results.mat'), results)

    # plt.figure(figsize=(8, 6))  # 设置画布大小
    # ax = plt.axes(projection='3d')  # 设置三维轴
    # ax.scatter3D(surf_0[0][105:-105, 105:-105].flatten(), surf_0[1][105:-105, 105:-105].flatten(),
    #              surf_0[2][105:-105, 105:-105].flatten(), c=surf_0[2][105:-105, 105:-105].flatten())  # 三个数组对应三个维度（三个数组中的数一一对应）
    # # 显示图形
    # plt.show()

    # plt.imshow(surf_0[0]*mask)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(surf_0[1]*mask)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(surf_0[2]*mask)
    # plt.colorbar()
    # plt.show()


def error_summary(mat_list_1, mat_list_2, mask, savemat_dir=''):
    mae_list = []
    results_mat = {}
    for i in range(len(mat_list_1)):
        tmp_mae = np.sum(np.abs((mat_list_1[i] - mat_list_2[i]) * mask)) / np.sum(mask)
        mae_list.append(tmp_mae)
        results_mat.update({'targetuvw_' + str(i): mat_list_1[i] * mask, 'labeluvw_' + str(i): mat_list_2[i] * mask,
                                  'erroruvw_' + str(i): (mat_list_1[i] - mat_list_2[i]) * mask})
    if savemat_dir != '':
        savemat(savemat_dir, results_mat)
    return mae_list


if __name__ == '__main__':
    # case_id = 3
    # uvw_ncorr, surf_0_ncorr, surf_1_ncorr, mask_ncorr, flow_ncorr, disparity_0_ncorr = ncorr_3D(case_id)
    # uvw_net, surf_0_net, surf_1_net, mask_net, flow_net, disparity_0_net = evaluate_net_case(case_id)
    # mask = mask_net * mask_ncorr
    #
    # save_dir = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\StereoEXP\DICe\results'
    #
    # uvw_mae = error_summary(uvw_net, uvw_ncorr, mask,) # savemat_dir=os.path.join(save_dir, str(case_id)+'_uvw_compare.mat')
    # surf_0_mae = error_summary(surf_0_net, surf_0_ncorr, mask) #, savemat_dir=os.path.join(save_dir, str(case_id)+'_surf0_compare.mat')
    # surf_1_mae = error_summary(surf_1_net, surf_1_ncorr, mask) #, savemat_dir=os.path.join(save_dir, str(case_id)+'_surf1_compare.mat')
    #
    # pixel_01_mae = error_summary(flow_ncorr, flow_net, mask,) # savemat_dir=os.path.join(save_dir, str(case_id)+'_uvw_compare.mat')
    # pixel_lr_mae = error_summary([disparity_0_ncorr], [disparity_0_net], mask,) # savemat_dir=os.path.join(save_dir, str(case_id)+'_uvw_compare.mat')
    #
    # print(uvw_mae, surf_0_mae, surf_1_mae)
    # print(pixel_01_mae, pixel_lr_mae)

    # evaluate_net_case_composite_single()
    evaluate_net_case_composite()