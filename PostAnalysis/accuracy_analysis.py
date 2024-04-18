import os.path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from eval import StrainNet_LD
from scipy.io import savemat

ncorr_results_path = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr'
label_path = r'E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\Valid'

def load_ncorr_mat_result(fname):
    if fname.endswith('.mat'):
        fname = os.path.join(ncorr_results_path, fname)
        tmp_ncorr_mat = h5py.File(fname)
        tmp_displacement_u = tmp_ncorr_mat['data_dic_save']['displacements']['plot_u_ref_formatted'][:].T.astype('float32')
        tmp_displacement_v = tmp_ncorr_mat['data_dic_save']['displacements']['plot_v_ref_formatted'][:].T.astype('float32')
        tmp_strain_exx = tmp_ncorr_mat['data_dic_save']['strains']['plot_exx_ref_formatted'][:].T.astype('float32')
        tmp_strain_exy = tmp_ncorr_mat['data_dic_save']['strains']['plot_exy_ref_formatted'][:].T.astype('float32')
        tmp_strain_eyy = tmp_ncorr_mat['data_dic_save']['strains']['plot_eyy_ref_formatted'][:].T.astype('float32')
    else:
        fname = os.path.join(ncorr_results_path, fname)
        tmp_dict = np.load(fname, allow_pickle=True)
        tmp_displacement_u, tmp_displacement_v, tmp_strain_exx, tmp_strain_exy, tmp_strain_eyy = tmp_dict['v'],tmp_dict['u'], tmp_dict['eyy'], tmp_dict['exy'], tmp_dict['exx']
    mask_dice = (tmp_displacement_u != 0) * (tmp_displacement_v != 0) * (tmp_strain_exx != 0) * (tmp_strain_eyy != 0) * (tmp_strain_exy != 0)
    return tmp_displacement_u, tmp_displacement_v, tmp_strain_exx, tmp_strain_exy, tmp_strain_eyy, mask_dice


def evaluate_net_case(fname, iter=1, with_net=True, with_decomp=False):
    my_evaluator = StrainNet_LD(calculate_strain=True, params_dir='../params/', strain_with_net=with_net, device='cpu', strain_radius=12, strain_skip=1, raw_kernel_size=55, strain_decomp=with_decomp, num_iter=iter)

    all_channels = np.load(os.path.join(label_path, fname))
    mask = all_channels[0, :, :] != 0
    ref_img = all_channels[0, :, :]
    def_img = all_channels[1, :, :]
    u_net, v_net, [exx_net, exy_net, eyy_net] = my_evaluator.correlate_imgpair(ref_img, def_img, mask, raw=False)

    return u_net, v_net, exx_net, exy_net, eyy_net, mask


def load_label(fname):
    all_channels = np.load(os.path.join(label_path, fname))
    mask = all_channels[0, :, :] != 0
    displacements = all_channels[2:, :, :]
    tmp_displacement_v = displacements[0, :, :]
    tmp_displacement_u = displacements[1, :, :]
    padded_u = np.pad(tmp_displacement_u, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    padded_v = np.pad(tmp_displacement_v, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

    u_y = padded_u[1:-1, 1:-1] - padded_u[0:-2, 1:-1]
    u_x = padded_u[1:-1, 1:-1] - padded_u[1:-1, 0:-2]
    v_y = padded_v[1:-1, 1:-1] - padded_v[0:-2, 1:-1]
    v_x = padded_v[1:-1, 1:-1] - padded_v[1:-1, 0:-2]

    exx = u_x + 0.5 * (u_x ** 2 + v_x ** 2)
    exy = 0.5 * (u_y + v_x + u_x * u_y + v_x * v_y)
    eyy = v_y + 0.5 * (u_y ** 2 + v_y ** 2)

    return tmp_displacement_u.astype('float32'), tmp_displacement_v.astype('float32'), exx.astype('float32'), exy.astype('float32'), eyy.astype('float32'), mask


def show_img(mat_list, mask, vmax=None, vmin=None, title=None):
    num_x = len(mat_list)
    num_y = len(mat_list[0])
    plt.figure(figsize=(num_y*3, num_x*3+1))
    for i in range(num_x):
        for j in range(num_y):
            plt.subplot(num_x, num_y, i * num_y + j + 1)
            plt.imshow(mat_list[i][j] * mask, vmin=vmin[i] if vmin is not None else None, vmax=vmax[i] if vmin is not None else None)
            plt.colorbar()
    plt.title(title if title is not None else '')
    plt.show()


def error_summary(labels, target, mask, savemat_dir=''):
    total_mae = 0
    epe_error = np.zeros_like(mask).astype('float64')
    mae_points = 0
    mask_epe = np.ones_like(mask)
    results_mat = {}
    amp = 0
    max_error = 0
    for i in range(len(target)):
        mask_epe *= mask * (target[i] != 0)
        mae_points += np.sum(mask * (target[i] != 0))
    for i in range(len(labels)):
        tmp_label = labels[i]
        tmp_target = target[i]
        tmp_mask = mask #* (tmp_target != 0)
        total_mae += np.sum(np.abs(tmp_target - tmp_label) * tmp_mask)
        epe_error += (tmp_target - tmp_label)**2 * mask_epe
        amp = max(amp, np.max(np.abs(tmp_label* tmp_mask)))
        max_error = max(max_error, np.max(np.abs(tmp_target - tmp_label) * tmp_mask))
        # tmp_target[tmp_mask==0] = np.nan
        # tmp_label[tmp_mask==0] = np.nan
        # results_mat.update({'target_' + str(i): tmp_target, 'label_' + str(i): tmp_label,
        #                           'error_' + str(i): tmp_target - tmp_label})
    if savemat_dir != '':
        savemat(savemat_dir, results_mat)

    mae = total_mae / mae_points
    epe = np.sum(np.sqrt(epe_error)) / np.sum(mask_epe)
    print('MAE: %.5f'%(mae), '\t MEPE: %.5f'%(epe))
    # print('AMP: %.5f'%(amp), '\t MAX MAE: %.5f'%(max_error))
    return mae, epe


if __name__ == '__main__':
    # case_id = 1887#1197 #1481 #644 #1197 #1105 #690 #1481#644 #  1105
    # 问题分析：
    # case_ids = [1499]
    # 更好的： 1065，
    iter = 2
    with_net = True
    with_decomp = False
    case_ids = [21, 28, 221, 275, 625, 644, 681, 690, 733, 753, 841, 917, 938, 1065, 1105, 1197, 1481, 1499, 1683, 1827, 1887, 135, 136]
    accept = [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    accept = [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    # accept = [1, 1, 1]
    results_list = []
    save_dir = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ALL_MAT'
    i = 0
    for case_id in case_ids:
        if not accept[i]:
            i += 1
            continue
        u_label, v_label, exx_label, exy_label, eyy_label, mask_label = load_label(str(case_id) + '_img&disp.npy')
        try:
            u_dice, v_dice, exx_dice, exy_dice, eyy_dice, mask_dice = load_ncorr_mat_result('CASE_'+str(case_id) + '.mat')
        except:
            u_dice, v_dice, exx_dice, exy_dice, eyy_dice, mask_dice = load_ncorr_mat_result('CASE_'+str(case_id) + '.npz')
        u_net, v_net, exx_net, exy_net, eyy_net, mask_net = evaluate_net_case(str(case_id) + '_img&disp.npy', iter=iter, with_net=with_net, with_decomp=with_decomp)
        mask = mask_label * mask_dice * mask_net
        mask = mask * (np.abs(u_net - u_label) < 2.0) * (np.abs(v_net - v_label) < 2.0)
        mask_epsilon = mask * (np.abs(exx_label - exx_net) < 0.05) * (np.abs(exy_label - exy_net) < 0.05) * (np.abs(eyy_label - eyy_net) < 0.05)
        # show_img([[u_label, u_dice, u_label-u_dice], [v_label, v_dice, v_label-v_dice], [exx_label, exx_dice, exx_label-exx_dice], [exy_label, exy_dice, exy_label-exy_dice], [eyy_label, eyy_dice, eyy_label-eyy_dice]],
        #          mask=mask_epsilon,
        #          vmin=[np.min(u_label[2:-2, 2:-2]), np.min(v_label[2:-2, 2:-2]), np.min(exx_label[2:-2, 2:-2]), np.min(exy_label[2:-2, 2:-2]), np.min(eyy_label[2:-2, 2:-2])],
        #          vmax=[np.max(u_label[2:-2, 2:-2]), np.max(v_label[2:-2, 2:-2]), np.max(exx_label[2:-2, 2:-2]), np.max(exy_label[2:-2, 2:-2]), np.max(eyy_label[2:-2, 2:-2])],title = str(case_id))
        if accept[i]:
            show_img([[u_label, u_net, u_label-u_net], [v_label, v_net, v_label-v_net], [exx_label, exx_net, exx_label-exx_net], [exy_label, exy_net, exy_label-exy_net], [eyy_label, eyy_net, eyy_label-eyy_net]],
                     mask=mask_epsilon,
                     vmin=[np.min(u_label[2:-2, 2:-2]), np.min(v_label[2:-2, 2:-2]), np.min(exx_label[2:-2, 2:-2]), np.min(exy_label[2:-2, 2:-2]), np.min(eyy_label[2:-2, 2:-2])],
                     vmax=[np.max(u_label[2:-2, 2:-2]), np.max(v_label[2:-2, 2:-2]), np.max(exx_label[2:-2, 2:-2]), np.max(exy_label[2:-2, 2:-2]), np.max(eyy_label[2:-2, 2:-2])])
        print('NCORR/Net Displacement ERROR of Case: %d'%(case_id))
        mae_disp_dice, epe_disp_dice = error_summary(labels=[u_label, v_label], target=[u_dice, v_dice], mask=mask, ) #savemat_dir=os.path.join(save_dir, str(case_id)+'_uv_ncorr.mat')
        mae_disp_net, epe_disp_net = error_summary(labels=[u_label, v_label], target=[u_net, v_net], mask=mask, ) #savemat_dir=os.path.join(save_dir, str(case_id)+'_it'+str(iter)+'_uv_net.mat')
        print('NCORR/NET Strain ERROR of Case: %d'%(case_id))
        mae_strain_dice, epe_strain_dice = error_summary(labels=[exx_label, exy_label, eyy_label], target=[exx_dice, exy_dice, eyy_dice], mask=mask_epsilon, ) #savemat_dir=os.path.join(save_dir, str(case_id)+'_strain_ncorr.mat')
        mae_strain_net, epe_strain_net = error_summary(labels=[exx_label, exy_label, eyy_label], target=[exx_net, exy_net, eyy_net], mask=mask_epsilon, ) #savemat_dir=os.path.join(save_dir, str(case_id)+'_it'+str(iter)+'_net'+str(int(with_net))+'_strain.mat')
        results_list.append([mae_disp_dice, mae_disp_net, epe_disp_dice, epe_disp_net, mae_strain_dice, mae_strain_net, epe_strain_dice, epe_strain_net])
        i += 1
    results_array = np.array(results_list)
    # np.savetxt('accuracy_results_it'+str(iter)+'_net'+str(int(with_net))+'_decomp'+str(int(with_decomp))+'.csv', results_array, delimiter=',')



