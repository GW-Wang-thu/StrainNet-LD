import os

import numpy as np
import cv2
from eval import StrainNet_LD, StrainNet_2D_Autoscale
import time
import matplotlib.pyplot as plt
import open3d as o3d
import json


def render_results(img, flow, mask, alpha=0.5, min_flow=None, max_flow=None, color_map=None, strain=False, fname=''):
    '''img、flow同形，mask标注了哪些位置有flow，其余位置透明度为1'''
    if len(img.shape) == 2:
        original_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 确保读
    else:
        original_image = img

    # 假设我们有一个二维矩阵，这里用随机数模拟
    if max_flow is None:
        min_flow_real = np.min(flow)
        max_flow_real = np.max(flow)
        flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))
        max_flow_label = np.max(flow)
        min_flow_label = np.min(flow)
        # flow = 1 - flow

    else:
        flow = (flow - min_flow) * (flow > min_flow) + min_flow
        flow = (flow - max_flow) * (flow < max_flow) + max_flow
        flow = (flow - min_flow) / (max_flow - min_flow)
        min_flow_label = 0
        max_flow_label = 1
        min_flow_real = min_flow
        max_flow_real = max_flow

    # 生成伪彩图，这里使用jet颜色映射作为示例
    pseudo_color_image = cv2.applyColorMap((flow * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # 生成colormap
    if color_map is None:
        color_map = np.ones(shape=(788, 190, 3), dtype=np.uint8) * 255
        colorbar = cv2.applyColorMap(np.expand_dims(np.linspace(max_flow_label*255, min_flow_label*255, num=768), 1).repeat(30, axis=1).astype(np.uint8), cv2.COLORMAP_JET)
        color_map[10:-10, 25:55, :3] = colorbar
        if strain:
            cv2.putText(color_map, str(int(max_flow_real))+'e-3', (35+25, 35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
            cv2.putText(color_map, str(int(min_flow_real))+'e-3', (35+25, 768), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
            cv2.putText(color_map, str(int((min_flow_real+max_flow_real)/2))+'e-3', (35+25, 768//2+17), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
        else:
            cv2.putText(color_map, str(int(max_flow_real))+' Pix', (35+25, 35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
            cv2.putText(color_map, str(int(min_flow_real))+' Pix', (35+25, 768), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
            cv2.putText(color_map, str(int((min_flow_real+max_flow_real)/2))+' Pix', (35+25, 768//2+17), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)

        color_map = cv2.resize(color_map, dsize=(int(img.shape[0]*190/788), img.shape[0]), interpolation=cv2.INTER_CUBIC)

    alpha_channel = np.ones(shape=pseudo_color_image.shape[:2]) * alpha * mask
    overlay_image = cv2.resize(pseudo_color_image, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 计算混合后的图像
    blended_image = np.zeros(shape=(original_image.shape[0], original_image.shape[1]+color_map.shape[1], 3), dtype='uint8')
    blended_image[:, original_image.shape[1]:original_image.shape[1]+color_map.shape[1], :] = color_map[:, :, :3]
    for c in range(3):  # 对BGR的每个通道进行操作
        blended_image[:, :original_image.shape[1], c] = (1 - alpha_channel) * original_image[:, :, c] + alpha_channel * overlay_image[:, :, c]

    # 显示结果
    if fname != '':
        print(fname)
        cv2.imwrite(fname, cv2.resize(blended_image, dsize=None, fx=900/max(img.shape), fy=900/max(img.shape), interpolation=cv2.INTER_CUBIC))
    cv2.imshow('Blended Image', cv2.resize(blended_image, dsize=None, fx=900/max(img.shape), fy=900/max(img.shape), interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(10)
    # cv2.destroyAllWindows()

#
def render_results_3D(xyz_array, rgb_array, mask=None, flow=None, alpha=0.5, max_flow=None, color_map=None):
    '''img、flow同形，mask标注了哪些位置有flow，其余位置透明度为1, 数组的形状为(*, 3)'''
    if flow is not None:
        if max_flow is None:
            min_flow_real = np.min(flow)
            max_flow_real = np.max(flow)
            flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))
            max_flow_label = np.max(flow)
            min_flow_label = np.min(flow)
            # flow = 1 - flow
        else:
            flow = (flow + max_flow) / (2 * max_flow)
            min_flow_label = 0
            max_flow_label = 1
            min_flow_real = -max_flow
            max_flow_real = max_flow
        flow_color = cv2.applyColorMap((flow * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, 0, :]
        alpha_channel = np.ones(shape=flow_color.shape[0]) * alpha * mask
        blended_color = np.zeros(shape=(flow_color.shape[0], 3), dtype='uint8')
        for c in range(3):  # 对BGR的每个通道进行操作
            blended_color[:, c] = (1 - alpha_channel) * rgb_array[:, c] + alpha_channel * flow_color[:, c]
        rgb_array = blended_color


    # 生成colormap
    if color_map is None and flow is not None:
        color_map = np.ones(shape=(788, 190, 3), dtype=np.uint8) * 255
        colorbar = cv2.applyColorMap(np.expand_dims(np.linspace(max_flow_label*255, min_flow_label*255, num=768), 1).repeat(30, axis=1).astype(np.uint8), cv2.COLORMAP_JET)
        color_map[10:-10, 25:55, :3] = colorbar
        cv2.putText(color_map, str(int(max_flow_real))+' Pix', (35+25, 35), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
        cv2.putText(color_map, str(int(min_flow_real))+' Pix', (35+25, 768), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
        cv2.putText(color_map, str(int((min_flow_real+max_flow_real)/2))+' Pix', (35+25, 768//2+17), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
        color_map = cv2.resize(color_map, dsize=(190, 768), interpolation=cv2.INTER_CUBIC)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)
    pcd.colors = o3d.utility.Vector3dVector(rgb_array / 255.0)
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 2  # 设置点的大小
    render_option.background_color = [1, 1, 1]  # 设置背景颜色为白色
    vis.run()


class Seq_precessor_3D:
    def __init__(self, params_file="./params/Calib.json", alpha=0.5, view='2D', blockpos=None):
        self.alpha = alpha

        with open(params_file, "r") as f:
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

        self.block_pos = blockpos

        self.shift_LY, self.shift_LX = np.meshgrid(range(0, params['LResolution'][1]), range(0, params['LResolution'][0]))
        if blockpos is not None:
            self.shift_LX = self.shift_LX[blockpos[0]:blockpos[1], blockpos[2]:blockpos[3]]
            self.shift_LY = self.shift_LY[blockpos[0]:blockpos[1], blockpos[2]:blockpos[3]]

        self.correlator = StrainNet_LD(device='cuda', calculate_strain=False, strain_with_net=False, num_iter=2)

        if view == '3D':
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="3D Points")
            self.vis.get_render_option().point_size = 3
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[10, 10, 500])
            self.vis.add_geometry(axis_pcd)
            # 获取当前视图控制对象
            view_control = self.vis.get_view_control()
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(params['LResolution'][1], params['LResolution'][0], self.fx_l, self.fy_l, self.cx_l, self.cy_l)

            # Set the camera parameters
            camera_params = o3d.camera.PinholeCameraParameters()
            camera_params.intrinsic = intrinsic
            extrinsic = np.eye(4)
            extrinsic[3, 3] = 400
            camera_params.extrinsic = extrinsic
            view_control.convert_from_pinhole_camera_parameters(camera_params, True)

            # 更新渲染器
            self.vis.update_renderer()

    def calc_render_imgs_3d_view(self, ref_img_l, ref_img_r, cur_img_list_l, cur_img_list_r, img_box, mask):
        '''以曲面高度为colorbar展示'''
        self.correlator.update_refimg(ref_img_l, mask)
        disparity_x, _, _ = self.correlator.correlate_sequence(ref_img_r)
        tmp_surf_coord = self.__recon_3d__(disparity_x, box=img_box, mask=mask)
        ref_img_color = cv2.cvtColor(ref_img_l, cv2.COLOR_GRAY2BGR)
        ref_surf_rgb = np.array([ref_img_color[:, :, 2].flatten(), ref_img_color[:, :, 1].flatten(), ref_img_color[:, :, 0].flatten()]).swapaxes(0, 1)
        tmp_fused_color = self.__fuse_color__(rgb_array=ref_surf_rgb, flow=tmp_surf_coord[2, :, :].flatten(), mask=mask.flatten())  # 以曲面高度为 colorbar

        tmp_surf_coord = tmp_surf_coord.reshape([3, -1]).swapaxes(0, 1)
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(tmp_surf_coord)
        tmp_pcd.colors = o3d.utility.Vector3dVector(tmp_fused_color)
        # tmp_pcd.points = o3d.utility.Vector3dVector(np.random.uniform(-1, 1, size=(10, 3)))
        # tmp_pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(10, 3)))
        self.vis.add_geometry(tmp_pcd)
        i = 0
        while True:
            tmp_l_img = cur_img_list_l[i]
            tmp_r_img = cur_img_list_r[i]
            # 计算参考帧到当前帧的光流
            flow_x, flow_y, _ = self.correlator.correlate_sequence(tmp_l_img)
            # 计算下一帧视差视差
            disparity_x, _, _ = self.correlator.correlate_imgpair(tmp_l_img, tmp_r_img, mask)
            # 利用光流和视差生成新的世界坐标
            non_zero_index = np.where(mask.ravel() != 0)
            fr, fc = np.unravel_index(non_zero_index[0][0], mask.shape)
            lr, lc = np.unravel_index(non_zero_index[0][-1], mask.shape)
            tmp_surf_coord = self.__recon_3d__(disparity_x, flow_x, flow_y, box=img_box, mask=mask)
            tmp_fused_color = self.__fuse_color__(rgb_array=ref_surf_rgb, flow=tmp_surf_coord[2, :, :].flatten(), mask=mask.flatten())  # 以曲面高度为 colorbar

            # 给点云添加显示的数据
            tmp_surf_coord = tmp_surf_coord.reshape([3, -1]).swapaxes(0, 1)
            tmp_pcd.points = o3d.utility.Vector3dVector(tmp_surf_coord)  # 更新点云坐标位置
            tmp_pcd.colors = o3d.utility.Vector3dVector(tmp_fused_color)  # 更新点云的颜色
            # tmp_pcd.points = o3d.utility.Vector3dVector(np.random.uniform(-1, 1, size=(10, 3)))
            # tmp_pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(10, 3)))
            # tmp_pcd = tmp_pcd.voxel_down_sample(voxel_size=10)
            # update_renderer显示当前的数据
            self.vis.update_geometry(tmp_pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            # cv2.waitKey(10000)
            i += 1

    def calc_render_imgs_2d_view_surf(self, cur_img_list_l, cur_img_list_r, mask=None, demo_img_list=None, demo_pos=None):
        '''以曲面高度为colorbar展示'''

        for i in range(len(cur_img_list_l)):
            L0 = cur_img_list_l[i]
            R0 = cur_img_list_r[i]
            imsize = L0.shape
            if demo_img_list is not None:
                mask_demo = np.zeros_like(demo_img_list[i])
                mask_demo[demo_pos[0]:demo_pos[0]+imsize[0], demo_pos[1]:demo_pos[1]+imsize[1]] = 1
                tmp_Demo_img = demo_img_list[i]
                if mask is not None:
                    mask_demo[demo_pos[0]:demo_pos[0]+imsize[0], demo_pos[1]:demo_pos[1]+imsize[1]] = mask
            else:
                tmp_Demo_img = L0
                if mask is not None:
                    mask_demo = mask
                else:
                    mask_demo = np.ones_like(tmp_Demo_img)
            # 计算视差
            L0 = cv2.resize(L0, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            R0 = cv2.resize(R0, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
            if mask is not None:
                mask = cv2.resize(mask, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            disparity_x, _, _ = self.correlator.correlate_imgpair(L0, R0, mask=mask)
            rescale_factor = (imsize[0] / 1024, imsize[1] / 1024)
            disparity_x = cv2.resize(disparity_x * rescale_factor[1], dsize=imsize, interpolation=cv2.INTER_CUBIC)

            # 利用视差生成世界坐标
            tmp_surf_coord = self.__recon_3d__(disparity_x, mask=mask)
            color_map = tmp_surf_coord[2]
            if demo_img_list is not None:
                color_map_demo = np.zeros_like(tmp_Demo_img)
                color_map_demo[demo_pos[0]:demo_pos[0]+imsize[0], demo_pos[1]:demo_pos[1]+imsize[1]] = color_map
            else:
                color_map_demo = color_map

            tmp_fused_img = self.__fuse_color_2d__(img=tmp_Demo_img, flow_x=0, flow_y=0, max_value=460, min_value=440,
                                                   color_value=color_map_demo, mask=mask_demo, gen_colormap=True)  # 以曲面高度为 colorbar

            cv2.imshow('2D Recon Height', cv2.resize(tmp_fused_img, dsize=(1224, 1024)))
            cv2.waitKey(10)

    def calc_render_imgs_2d_view_flow(self, ref_img_l, ref_img_r, cur_img_list_l, cur_img_list_r, img_box, mask, view="Eulerian"):
        '''以曲面高度为colorbar展示'''
        self.correlator.update_refimg(ref_img_l, mask)
        disparity_x, _, _ = self.correlator.correlate_sequence(ref_img_r)
        tmp_surf_coord = self.__recon_3d__(disparity_x, box=img_box, mask=mask)
        ref_img_color = cv2.cvtColor(ref_img_l, cv2.COLOR_GRAY2BGR)
        tmp_fused_img = self.__fuse_color_2d__(img=ref_img_color, flow_x=0, flow_y=0, color_value=np.zeros_like(mask), mask=mask, gen_colormap=True)
        cv2.imshow('2D Recon Flow', tmp_fused_img)
        cv2.waitKey(10)

        for i in range(len(cur_img_list_l)):
            tmp_l_img = cur_img_list_l[i]
            tmp_r_img = cur_img_list_r[i]
            # 计算参考帧到当前帧的光流
            flow_x, flow_y, _ = self.correlator.correlate_sequence(tmp_l_img)
            # 计算下一帧视差视差
            disparity_x, _, _ = self.correlator.correlate_imgpair(tmp_l_img, tmp_r_img, mask)
            # 利用光流和视差生成新的世界坐标
            tmp_surf_coord_n = self.__recon_3d__(disparity_x, flow_x, flow_y, box=img_box, mask=mask)

            tmp_flow = tmp_surf_coord_n - tmp_surf_coord
            tmp_fused_img = self.__fuse_color_2d__(img=tmp_l_img, flow_x=flow_x, flow_y=flow_y, color_value=tmp_flow[2, :, :], mask=mask, gen_colormap=True)  # 以曲面高度为 colorbar

            cv2.imshow('2D Recon Flow', tmp_fused_img)
            cv2.waitKey(10)
            tmp_surf_coord = tmp_surf_coord_n

    def __recon_3d__(self, disparity_x, flow_x=0, flow_y=0, mask=None):

        left_coord_y_1 = self.shift_LX + flow_x
        left_coord_x_1 = self.shift_LY + flow_y

        right_coord_x_1 = left_coord_x_1 + disparity_x

        if mask is not None:
            xw_pad = np.zeros_like(disparity_x)
            yw_pad = np.zeros_like(disparity_x)
            zw_pad = np.zeros_like(disparity_x)
            non_zero_index = np.where(mask.ravel() != 0)
            fr, fc = np.unravel_index(non_zero_index[0][0], mask.shape)
            lr, lc = np.unravel_index(non_zero_index[0][-1], mask.shape)
            left_coord_x_1 = left_coord_x_1[fr:lr, fc:lc]
            left_coord_y_1 = left_coord_y_1[fr:lr, fc:lc]
            right_coord_x_1 = right_coord_x_1[fr:lr, fc:lc]

        zw = (self.Tx_r - self.Tz_r * (right_coord_x_1 + 1.0 - self.cx_r) / self.fx_r) / \
               (((right_coord_x_1 + 1.0 - self.cx_r) / self.fx_r) * (
                       self.RRot[2, 0] * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l + self.RRot[2, 1] * (
                       left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l + self.RRot[2, 2]) -
                (self.RRot[0, 0] * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l + self.RRot[0, 1] * (
                        left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l +
                 self.RRot[0, 2]))
        xw = zw * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l
        yw = zw * (left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l
        if mask is not None:
            xw_pad[fr:lr, fc:lc] = xw
            yw_pad[fr:lr, fc:lc] = yw
            zw_pad[fr:lr, fc:lc] = zw
            xw, yw, zw = xw_pad, yw_pad, zw_pad
        return np.array([xw, yw, zw])

    def __fuse_color__(self, rgb_array, flow, max_flow=None, mask=1, gen_colormap=False):
        if flow is not None:
            if max_flow is None:
                min_flow_real = np.min(flow)
                max_flow_real = np.max(flow)
                flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))
                max_flow_label = np.max(flow)
                min_flow_label = np.min(flow)
                # flow = 1 - flow
            else:
                flow = (flow + max_flow) / (2 * max_flow)
                min_flow_label = 0
                max_flow_label = 1
                min_flow_real = -max_flow
                max_flow_real = max_flow
            flow_color = cv2.applyColorMap((flow * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, 0, :]
            alpha_channel = np.ones(shape=flow_color.shape[0]) * self.alpha * mask
            blended_color = np.zeros(shape=(flow_color.shape[0], 3), dtype='uint8')
            for c in range(3):  # 对BGR的每个通道进行操作
                blended_color[:, c] = (1 - alpha_channel) * rgb_array[:, c] + alpha_channel * flow_color[:, c]
            rgb_array = blended_color

        # 生成colormap
        if gen_colormap:
            color_map = np.ones(shape=(788, 190, 3), dtype=np.uint8) * 255
            colorbar = cv2.applyColorMap(
                np.expand_dims(np.linspace(max_flow_label * 255, min_flow_label * 255, num=768), 1).repeat(30, axis=1).astype(
                    np.uint8), cv2.COLORMAP_JET)
            color_map[10:-10, 25:55, :3] = colorbar
            cv2.putText(color_map, str(int(max_flow_real)) + ' Pix', (35 + 25, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(30, 30, 30), thickness=2)
            cv2.putText(color_map, str(int(min_flow_real)) + ' Pix', (35 + 25, 768), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(30, 30, 30), thickness=2)
            cv2.putText(color_map, str(int((min_flow_real + max_flow_real) / 2)) + ' Pix', (35 + 25, 768 // 2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
            color_map = cv2.resize(color_map, dsize=(190, 768), interpolation=cv2.INTER_CUBIC)
            rgb_array = np.concatenate([rgb_array, color_map], axis=0)
        # rgb_array[mask==0] = 1

        return rgb_array / 255.0

    def __fuse_color_2d__(self, img, flow_x, flow_y, color_value, max_value=None, min_value=None, mask=1, gen_colormap=False):
        img_grid_y, img_grid_x = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
        color_value_init = color_value

        color_value_eulerian = np.zeros_like(color_value, dtype='float32')
        x_pos, y_pos = np.round(img_grid_x + flow_x).astype('int32'), np.round(img_grid_y + flow_y).astype('int32')
        x_pos = x_pos * (x_pos >= 0) * mask
        x_pos = (x_pos - x_pos.shape[0] + 1) * (x_pos < x_pos.shape[0]) + x_pos.shape[0] - 1
        y_pos = y_pos * (y_pos >= 0) * mask
        y_pos = (y_pos - y_pos.shape[1] + 1) * (y_pos < y_pos.shape[1]) + y_pos.shape[1] - 1
        color_value_eulerian[(x_pos.view(), y_pos.view())] = color_value.view()
        # mask_eulerian = np.zeros_like(color_value)
        # mask_eulerian[(x_pos.view(), y_pos.view())] = mask.view()
        mask_eulerian = (color_value_eulerian != 0)
        color_value = color_value_eulerian

        # 假设我们有一个二维矩阵，这里用随机数模拟
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 确保读
        if max_value is None:
            min_flow_real = np.min(color_value_init)
            max_flow_real = np.max(color_value_init)
            color_value = (color_value - min_flow_real) / (max_flow_real - min_flow_real)
            print(min_flow_real, max_flow_real)
            color_value = (color_value >= 0) * color_value
            color_value = (color_value - 1.0) * (color_value < 1.0) + 1.0
            max_flow_label = np.max(color_value)
            min_flow_label = np.min(color_value)
            # flow = 1 - flow
        else:
            color_value = (color_value - min_value) / (max_value - min_value)
            min_flow_label = 0
            max_flow_label = 1
            min_flow_real = min_value
            max_flow_real = max_value

        # 生成伪彩图，这里使用jet颜色映射作为示例
        pseudo_color_image = cv2.applyColorMap((color_value * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # 生成colormap
        if gen_colormap:
            color_map = np.ones(shape=(788, 190, 3), dtype=np.uint8) * 255
            colorbar = cv2.applyColorMap(np.expand_dims(np.linspace(max_flow_label * 255, min_flow_label * 255, num=768), 1).repeat(30, axis=1).astype(np.uint8), cv2.COLORMAP_JET)
            color_map[10:-10, 25:55, :3] = colorbar
            cv2.putText(color_map, str(int(max_flow_real)) + ' mm', (35 + 25, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(30, 30, 30), thickness=2)
            cv2.putText(color_map, str(int(min_flow_real)) + ' mm', (35 + 25, 768), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(30, 30, 30), thickness=2)
            cv2.putText(color_map, str(int((min_flow_real + max_flow_real) / 2)) + ' mm', (35 + 25, 768 // 2 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(30, 30, 30), thickness=2)
            color_map = cv2.resize(color_map, dsize=(int(img.shape[0] * 190 / 788), img.shape[0]),
                                   interpolation=cv2.INTER_CUBIC)

        alpha_channel = np.ones(shape=pseudo_color_image.shape[:2]) * self.alpha * mask_eulerian
        overlay_image = cv2.resize(pseudo_color_image, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        # 计算混合后的图像
        blended_image = np.zeros(shape=(img.shape[0], img.shape[1] + color_map.shape[1], 3),
                                 dtype='uint8')
        blended_image[:, img.shape[1]:img.shape[1] + color_map.shape[1], :] = color_map[:, :, :3]
        for c in range(3):  # 对BGR的每个通道进行操作
            blended_image[:, :img.shape[1], c] = (1 - alpha_channel) * img[:, :, c] + alpha_channel * overlay_image[:, :, c]
        return blended_image


def seq_processing_2D(view='Lagrangian'):
    dataset_dir = r'E:\Data\Experiments\LargeDeformation\20221014\MV-CE200-10UM (00F89350571)'
    all_imgs = os.listdir(dataset_dir)
    all_imgs = [os.path.join(dataset_dir, f) for f in all_imgs]
    ref_img_id = 0
    def_img_seq = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # mask_img_ref = [200, 1960, 2100, 3470]
    # img_block = [100, 3500, 1980, 3570]
    # mask_img_ref = [200, 1960, 2100, 3470]
    mask_img_ref = [200-35, 1920+20, 2125, 3460]  # 11
    img_block = [100, 3500, 1980, 3570]
    resize_factor = (0.25, 0.5)
    ds_factor = 128
    ref_img_init = cv2.imread(all_imgs[ref_img_id], cv2.IMREAD_GRAYSCALE)[img_block[0]:img_block[0]+ds_factor*((img_block[1]-img_block[0])//ds_factor+1),
                            img_block[2]:img_block[2]+ds_factor*((img_block[3]-img_block[2])//ds_factor+1)]

    ref_img = cv2.resize(ref_img_init, dsize=None, fx=resize_factor[1], fy=resize_factor[0], interpolation=cv2.INTER_CUBIC)
    def_img_list = []
    for img_id in def_img_seq:
        tmp_img = cv2.imread(all_imgs[img_id], cv2.IMREAD_GRAYSCALE)[img_block[0]:img_block[0]+ds_factor*((img_block[1]-img_block[0])//ds_factor+1),
                            img_block[2]:img_block[2]+ds_factor*((img_block[3]-img_block[2])//ds_factor+1)]
        # cv2.imwrite(all_imgs[img_id][:-4]+'_crop_'+str(img_id)+'.bmp', tmp_img)
        def_img_list.append(cv2.resize(tmp_img, dsize=None, fx=resize_factor[1], fy=resize_factor[0], interpolation=cv2.INTER_CUBIC))
    img_grid_y, img_grid_x = np.meshgrid(range(def_img_list[-1].shape[1]), range(def_img_list[-1].shape[0]))
    img_grid_x = img_grid_x[int((mask_img_ref[0]-img_block[0])*resize_factor[0]):int((mask_img_ref[1]-img_block[0])*resize_factor[0]),
                            int((mask_img_ref[2]-img_block[2])*resize_factor[1]):int((mask_img_ref[3]-img_block[2])*resize_factor[1])]
    img_grid_y = img_grid_y[int((mask_img_ref[0]-img_block[0])*resize_factor[0]):int((mask_img_ref[1]-img_block[0])*resize_factor[0]),
                            int((mask_img_ref[2]-img_block[2])*resize_factor[1]):int((mask_img_ref[3]-img_block[2])*resize_factor[1])]
    mask = np.zeros_like(ref_img_init)
    mask[mask_img_ref[0]-img_block[0]:mask_img_ref[1]-img_block[0], mask_img_ref[2]-img_block[2]:mask_img_ref[3]-img_block[2]] = 1
    mask = cv2.resize(mask, dsize=None, fx=resize_factor[1], fy=resize_factor[0], interpolation=cv2.INTER_NEAREST)

    my_correlator = StrainNet_LD(device='cuda', calculate_strain=True, strain_with_net=False, num_iter=2, strain_radius=13, strain_skip=1,light_adjust=False)
    my_correlator.update_refimg(ref_img, mask)
    max_flow = 0.03
    min_flow = -0.03
    for i in range(len(def_img_list)):
        now = time.perf_counter()
        v, u, strain = my_correlator.correlate_sequence(def_img_list[i], scales=(0.5, 0.3))
        tmp_fname = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\RUBBEREXP\\' + str(def_img_seq[i]) + '_uv.npy'
        # np.save(tmp_fname, np.stack([u, v]))
        # np.save(tmp_fname[:-5]+'strain.npy', np.stack(strain))
        if view == 'Lagrangian':
            render_results(img=cv2.resize(ref_img, dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_LINEAR),
                           flow=cv2.resize(u / resize_factor[0], dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_LINEAR),
                           mask=cv2.resize(mask, dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_LINEAR))
        else:
            u = u[int((mask_img_ref[0]-img_block[0])*resize_factor[0]):int((mask_img_ref[1]-img_block[0])*resize_factor[0]),
                  int((mask_img_ref[2]-img_block[2])*resize_factor[1]):int((mask_img_ref[3]-img_block[2])*resize_factor[1])]
            v = v[int((mask_img_ref[0]-img_block[0])*resize_factor[0]):int((mask_img_ref[1]-img_block[0])*resize_factor[0]),
                  int((mask_img_ref[2]-img_block[2])*resize_factor[1]):int((mask_img_ref[3]-img_block[2])*resize_factor[1])]
            flow_Eulerian_u = np.zeros_like(def_img_list[i], dtype='float32')
            flow_Eulerian_v = np.zeros_like(def_img_list[i], dtype='float32')
            flow_Eulerian_uv = np.zeros_like(def_img_list[i], dtype='float32')
            x_pos, y_pos = np.round(img_grid_x+u).astype('int32'), np.round(img_grid_y+v).astype('int32')

            flow_Eulerian_u[(x_pos.view(), y_pos.view())] = u.view() / resize_factor[0]
            flow_Eulerian_v[(x_pos.view(), y_pos.view())] = v.view() / resize_factor[1]
            mask = (flow_Eulerian_v != 0).astype('float32')
            render_results(img=cv2.resize(def_img_list[i], dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_LINEAR),
                           flow=cv2.resize(flow_Eulerian_v , dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_NEAREST),
                           mask=cv2.resize(mask, dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_NEAREST)>0,
                           alpha=0.8, fname=tmp_fname[:-4]+'.bmp')

            mask_demo = np.zeros_like(strain[0])
            mask_demo[1, 1] = 1.0
            strain_xx = strain[0]
            strain_xy = strain[1]
            strain_yy = strain[2]
            strain_xx = strain_xx[int((mask_img_ref[0]-img_block[0])*resize_factor[0]):int((mask_img_ref[1]-img_block[0])*resize_factor[0]),
                                  int((mask_img_ref[2]-img_block[2])*resize_factor[1]):int((mask_img_ref[3]-img_block[2])*resize_factor[1])]
            strain_yy = strain_yy[int((mask_img_ref[0]-img_block[0])*resize_factor[0]):int((mask_img_ref[1]-img_block[0])*resize_factor[0]),
                                  int((mask_img_ref[2]-img_block[2])*resize_factor[1]):int((mask_img_ref[3]-img_block[2])*resize_factor[1])]
            strain_xy = strain_xy[int((mask_img_ref[0]-img_block[0])*resize_factor[0]):int((mask_img_ref[1]-img_block[0])*resize_factor[0]),
                                  int((mask_img_ref[2]-img_block[2])*resize_factor[1]):int((mask_img_ref[3]-img_block[2])*resize_factor[1])]

            mask_demo = np.zeros_like(strain_xx)
            mask_demo[:-20, 15:-15] = 1.0
            strain_xx = strain_xx * mask_demo
            strain_xy = strain_xy * mask_demo
            strain_yy = strain_yy * mask_demo

            flow_Eulerian_u[(x_pos.view(), y_pos.view())] = strain_xx.view() * 1000
            flow_Eulerian_v[(x_pos.view(), y_pos.view())] = strain_yy.view() * 700
            flow_Eulerian_uv[(x_pos.view(), y_pos.view())] = strain_xy.view() * 1000

            # mask = (flow_Eulerian_uv != 0).astype('float32')
            # plt.show()
            # render_results(img=cv2.resize(def_img_list[i], dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_LINEAR),
            #                flow=cv2.resize(flow_Eulerian_uv, dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_NEAREST),
            #                mask=cv2.resize(mask, dsize=None, fx=1/resize_factor[1], fy=1/resize_factor[0], interpolation=cv2.INTER_NEAREST)>0,
            #                max_flow=max_flow * 1000, min_flow=min_flow * 1000, strain=True, alpha=0.8,fname=tmp_fname[:-6]+'s12.bmp')

        # print('time consume: ', time.perf_counter() - now)
        if i==0:
            cv2.waitKey(1000)


def seq_processing_2D_rescaleN(view='Lagrangian'):
    dataset_dir = r'E:\Data\Experiments\LargeDeformation\20221014\MV-CE200-10UM (00F89350571)'
    all_imgs = os.listdir(dataset_dir)
    all_imgs = [os.path.join(dataset_dir, f) for f in all_imgs]
    ref_img_id = 0
    bd = 60
    def_img_seq = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # mask_img_ref = [200, 1960, 2100, 3470]
    # img_block = [100, 3500, 1980, 3570]
    img_block_demo = [100, 3500, 1980, 3570]
    img_block_calc = [200-10-bd, 2385+bd, 2125-bd, 3450+bd]
    # mask_img_ref = [200-20-bd, 1920-10+bd, 2125-bd, 3450+bd]   # 22
    # mask_img_ref = [200-20-bd, 1920+25+bd, 2150-bd, 3400+bd]  # 12
    mask_img_ref = [200-35-bd, 1920+20+bd, 2125-bd, 3450+bd]  # 11
    mask_ref_crop_to_demo = [mask_img_ref[0]-img_block_demo[0], mask_img_ref[1]-img_block_demo[0],
                             mask_img_ref[2]-img_block_demo[2], mask_img_ref[3]-img_block_demo[2]]
    mask_ref_crop_to_calc = [mask_img_ref[0]-img_block_calc[0], mask_img_ref[1]-img_block_calc[0],
                             mask_img_ref[2]-img_block_calc[2], mask_img_ref[3]-img_block_calc[2]]

    img_block_calc2demp = [img_block_calc[0]-img_block_demo[0], img_block_calc[1]-img_block_demo[0],
                           img_block_calc[2]-img_block_demo[2], img_block_calc[3]-img_block_demo[2]]

    demo_imsize = (img_block_demo[1]-img_block_demo[0], img_block_demo[3]-img_block_demo[2])
    calc_imsize = (img_block_calc[1]-img_block_calc[0], img_block_calc[3]-img_block_calc[2])

    ref_img_demo = cv2.imread(all_imgs[ref_img_id], cv2.IMREAD_GRAYSCALE)[img_block_demo[0]:img_block_demo[1], img_block_demo[2]:img_block_demo[3]]
    ref_img_calc = ref_img_demo[img_block_calc2demp[0]:img_block_calc2demp[1], img_block_calc2demp[2]:img_block_calc2demp[3]]


    def_img_list = []
    for img_id in def_img_seq:
        tmp_img = cv2.imread(all_imgs[img_id], cv2.IMREAD_GRAYSCALE)[img_block_demo[0]:img_block_demo[1], img_block_demo[2]:img_block_demo[3]]
        # cv2.imwrite(all_imgs[img_id][:-4]+'_crop_'+str(img_id)+'.bmp', tmp_img)
        def_img_list.append(tmp_img)
    img_grid_y, img_grid_x = np.meshgrid(range(def_img_list[-1].shape[1]), range(def_img_list[-1].shape[0]))
    img_grid_x = img_grid_x[mask_ref_crop_to_demo[0]+bd:mask_ref_crop_to_demo[1]-bd,mask_ref_crop_to_demo[2]+bd:mask_ref_crop_to_demo[3]-bd]
    img_grid_y = img_grid_y[mask_ref_crop_to_demo[0]+bd:mask_ref_crop_to_demo[1]-bd,mask_ref_crop_to_demo[2]+bd:mask_ref_crop_to_demo[3]-bd]

    my_correlator = StrainNet_2D_Autoscale(device='cuda', calculate_strain=True, strain_with_net=False, strain_decomp=True, num_iter=2, strain_radius=12, strain_skip=2,light_adjust=False)
    max_flow = -0.02#0.16#0.030#
    min_flow = -0.09#0.11#-0.030#
    for i in range(len(def_img_list)):
        now = time.perf_counter()
        v, u, strain = my_correlator.calc_2DFlow(ref_img_calc, def_img_list[i][img_block_calc2demp[0]:img_block_calc2demp[1], img_block_calc2demp[2]:img_block_calc2demp[3]], dsize=(1152, 896))#, dsize=(1280, 832)
        tmp_fname = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\RUBBEREXP\\A_' + str(def_img_seq[i]) + '_uv.npy'
        # np.save(tmp_fname, np.stack([u, v]))
        # np.save(tmp_fname[:-5]+'strain.npy', np.stack(strain))
        if view == 'Lagrangian':
            tmp_flow = np.zeros(demo_imsize)
            tmp_flow[img_block_calc2demp[0]+bd:img_block_calc2demp[1]-bd,img_block_calc2demp[2]+bd:img_block_calc2demp[3]-bd] = u[bd:-bd, bd:-bd]
            render_results(img=ref_img_demo,
                           flow=tmp_flow,
                           mask=tmp_flow != 0)
        else:
            u = u[mask_ref_crop_to_calc[0]+bd:mask_ref_crop_to_calc[1]-bd, mask_ref_crop_to_calc[2]+bd:mask_ref_crop_to_calc[3]-bd]
            v = v[mask_ref_crop_to_calc[0]+bd:mask_ref_crop_to_calc[1]-bd, mask_ref_crop_to_calc[2]+bd:mask_ref_crop_to_calc[3]-bd]
            flow_Eulerian_u = np.zeros_like(def_img_list[i], dtype='float32')
            flow_Eulerian_v = np.zeros_like(def_img_list[i], dtype='float32')
            flow_Eulerian_uv = np.zeros_like(def_img_list[i], dtype='float32')
            x_pos, y_pos = np.round(img_grid_x+u).astype('int32'), np.round(img_grid_y+v).astype('int32')

            flow_Eulerian_u[(x_pos.view(), y_pos.view())] = u.view()
            flow_Eulerian_v[(x_pos.view(), y_pos.view())] = v.view()
            mask = ((flow_Eulerian_v) != 0).astype('float32')
            render_results(img=def_img_list[i],
                           flow=flow_Eulerian_v,
                           mask=mask,
                           alpha=0.8, fname=tmp_fname[:-4]+'.bmp')

            strain_xx = strain[0][mask_ref_crop_to_calc[0]+bd:mask_ref_crop_to_calc[1]-bd, mask_ref_crop_to_calc[2]+bd:mask_ref_crop_to_calc[3]-bd]
            strain_xy = strain[1][mask_ref_crop_to_calc[0]+bd:mask_ref_crop_to_calc[1]-bd, mask_ref_crop_to_calc[2]+bd:mask_ref_crop_to_calc[3]-bd]
            strain_yy = strain[2][mask_ref_crop_to_calc[0]+bd:mask_ref_crop_to_calc[1]-bd, mask_ref_crop_to_calc[2]+bd:mask_ref_crop_to_calc[3]-bd]

            # mask_demo = np.zeros_like(strain_xx)
            # mask_demo[:-20, 15:-15] = 1.0
            # strain_xx = strain_xx * mask_demo
            # strain_xy = strain_xy * mask_demo
            # strain_yy = strain_yy * mask_demo

            flow_Eulerian_u[(x_pos.view(), y_pos.view())] = strain_xx.view() * 1000
            flow_Eulerian_v[(x_pos.view(), y_pos.view())] = strain_yy.view() * 720
            flow_Eulerian_uv[(x_pos.view(), y_pos.view())] = strain_xy.view() * 1000

            mask = ((flow_Eulerian_uv) != 0).astype('float32')
            plt.show()
            render_results(img=def_img_list[i],
                           flow=flow_Eulerian_u,
                           mask=mask,
                           max_flow=max_flow * 1000, min_flow=min_flow * 1000, strain=True, alpha=0.8,fname=tmp_fname[:-6]+'s11.bmp')

        # print('time consume: ', time.perf_counter() - now)
        if i==0:
            cv2.waitKey(1000)


def seq_processing_3D():
    #
    # ref_idx = '04874645'
    # resize_factor = 0.375
    # mask_img_ref = [150, 2700, 950, 2600]
    # img_block = [0, 2900, 800, 2800]
    #
    ref_idx = '00433360'
    resize_factor = 0.375
    mask_img_ref = [550, 2500, 1050, 2600]
    img_block = [100, 2900, 750, 3000]
    # ref_idx = 2343058
    # resize_factor = 0.3
    # mask_img_ref = [200, 2800, 500, 3500]
    # img_block = [200, 2800, 500, 3500]

    # 准备图像
    dataset_dir = r'E:\Data\Experiments\LargeDeformation\20240316'
    ds_factor = 64
    all_imgs = os.listdir(dataset_dir)
    all_imgs_l = [os.path.join(dataset_dir, f) for f in all_imgs if f.endswith('_l.bmp')]
    all_imgs_r = [os.path.join(dataset_dir, f) for f in all_imgs if f.endswith('_r.bmp')]
    ref_img_name = os.path.join(dataset_dir,"IMG_"+ref_idx+"_l.bmp")
    ref_img_id = all_imgs_l.index(ref_img_name)
    mask_img_ref_resize = [int(v * resize_factor) for v in mask_img_ref]
    img_block_resize = [int(v * resize_factor) for v in img_block]
    ref_img_l = cv2.resize(cv2.imread(all_imgs_l[ref_img_id], cv2.IMREAD_GRAYSCALE), dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    ref_img_l = ref_img_l[img_block_resize[0]:img_block_resize[0]+ds_factor*((img_block_resize[1]-img_block_resize[0])//ds_factor+1),
                          img_block_resize[2]:img_block_resize[2]+ds_factor*((img_block_resize[3]-img_block_resize[2])//ds_factor+1)]

    ref_img_r = cv2.resize(cv2.imread(all_imgs_r[ref_img_id], cv2.IMREAD_GRAYSCALE), dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    ref_img_r = ref_img_r[img_block_resize[0]:img_block_resize[0]+ds_factor*((img_block_resize[1]-img_block_resize[0])//ds_factor+1),
                          img_block_resize[2]:img_block_resize[2]+ds_factor*((img_block_resize[3]-img_block_resize[2])//ds_factor+1)]

    cur_img_seq = range(ref_img_id+20)[-25:]
    all_cur_l_img = []
    all_cur_r_img = []
    for i in cur_img_seq:
        all_cur_l_img.append(cv2.resize(cv2.imread(all_imgs_l[i], cv2.IMREAD_GRAYSCALE), dsize=None,
                                        fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)[
                                            img_block_resize[0]:img_block_resize[0] + ds_factor * ((img_block_resize[1] - img_block_resize[0]) // ds_factor + 1),
                                            img_block_resize[2]:img_block_resize[2] + ds_factor * ((img_block_resize[3] - img_block_resize[2]) // ds_factor + 1)])
        all_cur_r_img.append(cv2.resize(cv2.imread(all_imgs_r[i], cv2.IMREAD_GRAYSCALE), dsize=None,
                                        fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)[
                                            img_block_resize[0]:img_block_resize[0] + ds_factor * ((img_block_resize[1] - img_block_resize[0]) // ds_factor + 1),
                                            img_block_resize[2]:img_block_resize[2] + ds_factor * ((img_block_resize[3] - img_block_resize[2]) // ds_factor + 1)])
    mask = np.zeros_like(ref_img_l)
    # mask = cv2.resize(mask, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
    img_block_resize_real = [img_block_resize[0], img_block_resize[0]+ref_img_l.shape[0], img_block_resize[2], img_block_resize[2]+ref_img_l.shape[1]]
    # mask[mask_img_ref_resize[0]-img_block_resize[0]:mask_img_ref_resize[1]-img_block_resize[0], mask_img_ref_resize[2]-img_block_resize[2]:mask_img_ref_resize[3]-img_block_resize[2]] = 1
    mask[mask_img_ref_resize[0]-img_block_resize_real[0]:mask_img_ref_resize[1]-img_block_resize_real[0], mask_img_ref_resize[2]-img_block_resize_real[2]:mask_img_ref_resize[3]-img_block_resize_real[2]] = 1
    my_calculator_rendor_3d = Seq_precessor_3D(params_file='./params/Calib.json', alpha=0.5, resize_factor=resize_factor, view='2D')
    my_calculator_rendor_3d.calc_render_imgs_2d_view_flow(ref_img_l, ref_img_r, all_cur_l_img, all_cur_r_img, img_block_resize_real, mask)
    # my_calculator_rendor_3d = Seq_precessor_3D(params_file='./params/Calib.json', alpha=0.5, resize_factor=resize_factor, view='3D')
    # my_calculator_rendor_3d.calc_render_imgs_3d_view(ref_img_l, ref_img_r, all_cur_l_img, all_cur_r_img, img_block_resize_real, mask)


def seq_processing_3D_recon():
    #

    img_block = [750, 2400, 600, 2250]

    dataset_dir = r'E:\Data\Experiments\LargeDeformation\20240401\speckleless'
    all_imgs = os.listdir(dataset_dir)
    all_imgpath_l = [os.path.join(dataset_dir, f) for f in all_imgs if f.endswith('_l.bmp')]
    all_imgpath_r = [os.path.join(dataset_dir, f) for f in all_imgs if f.endswith('_r.bmp')]

    all_imgs_l = [cv2.imread(f, cv2.IMREAD_GRAYSCALE)[img_block[0]:img_block[1], img_block[2]:img_block[3]] for f in all_imgpath_l]
    all_imgs_r = [cv2.imread(f, cv2.IMREAD_GRAYSCALE)[img_block[0]:img_block[1], img_block[2]:img_block[3]] for f in all_imgpath_r]
    all_imgs_demo = [cv2.imread(f, cv2.IMREAD_GRAYSCALE)[img_block[0]-300:img_block[1]+300, img_block[2]-100:img_block[3]+500] for f in all_imgpath_l]

    my_calculator_rendor_3d = Seq_precessor_3D(params_file='./params/Calib.json', alpha=0.7, view='2D', blockpos=img_block)
    my_calculator_rendor_3d.calc_render_imgs_2d_view_surf(all_imgs_l, all_imgs_r, demo_img_list=all_imgs_demo, demo_pos=(300, 100))
    #


# def rename(dir = r'E:\Data\Experiments\LargeDeformation\20240401\speckleless'):
#     all_imgs = os.listdir(dir)
#     all_imgs = [os.path.join(dir, f) for f in all_imgs if f.endswith('.bmp')]
#     for i in range(len(all_imgs)):
#         tmp_name_int = int(all_imgs[i].split('\\')[-1].split('_')[1])
#         tmp_name_nint = '%0*d' % (9, tmp_name_int)
#         tmp_name_list = all_imgs[i].split('\\')[:-1]+['IMG_'+tmp_name_nint+'_'+all_imgs[i].split('\\')[-1].split('_')[-1]]
#         new_name = os.path.join(*tmp_name_list)
#         print(all_imgs[i])
#         print(new_name)
#         img = cv2.imread(all_imgs[i], cv2.IMREAD_GRAYSCALE)
#         cv2.imwrite(new_name, img)


if __name__ == '__main__':
    seq_processing_2D(view='Eulerian')
    # seq_processing_2D_rescaleN(view='Eulerian')
    # seq_processing_3D()
    # seq_processing_3D_recon()
    # rename()



    # img = cv2.imread('debug/original.bmp', cv2.IMREAD_COLOR)
    # x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), [1])
    # xyz_array = np.array([x.flatten(), y.flatten(), z.flatten()]).swapaxes(0, 1)
    # rgb_array = np.array([img[:, :, 2].flatten(), img[:, :, 1].flatten(), img[:, :, 0].flatten()]).swapaxes(0, 1)
    # flow = np.expand_dims(np.linspace(-5, 5, rgb_array.shape[0]), -1)
    # render_results_3D(xyz_array, rgb_array, flow=flow, alpha=0.5, mask=1)



