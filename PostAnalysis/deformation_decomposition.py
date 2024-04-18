import os.path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import torch
import torchvision.transforms as tf

from eval import StrainNet_LD
from scipy.io import savemat

from utils import interpolation

from accuracy_analysis import error_summary, show_img
from utils.StrainFilter import StrainFilter

from networks.DifferentialNet import DifferNet_FIT_Inter


ncorr_results_path = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain'
label_path = r'E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\Valid'

def load_ncorr_mat_result(fname):
    if fname.endswith('.mat'):
        fname = os.path.join(ncorr_results_path, fname)
        tmp_ncorr_mat = h5py.File(fname)
        tmp_displacement_u = tmp_ncorr_mat['data_dic_save']['displacements']['plot_u_dic'][:].T.astype('float32')
        tmp_displacement_v = tmp_ncorr_mat['data_dic_save']['displacements']['plot_v_dic'][:].T.astype('float32')
        tmp_strain_exx = tmp_ncorr_mat['data_dic_save']['strains']['plot_exx_ref_formatted'][:].T.astype('float32')
        tmp_strain_exy = tmp_ncorr_mat['data_dic_save']['strains']['plot_exy_ref_formatted'][:].T.astype('float32')
        tmp_strain_eyy = tmp_ncorr_mat['data_dic_save']['strains']['plot_eyy_ref_formatted'][:].T.astype('float32')
        # tmp_strain_exx = None
        # tmp_strain_exy = None
        # tmp_strain_eyy = None
    else:
        fname = os.path.join(ncorr_results_path, fname)
        tmp_dict = np.load(fname, allow_pickle=True)
        tmp_displacement_u, tmp_displacement_v, tmp_strain_exx, tmp_strain_exy, tmp_strain_eyy = tmp_dict['v'],tmp_dict['u'], tmp_dict['eyy'], tmp_dict['exy'], tmp_dict['exx']
    mask_dice = (tmp_displacement_u != 0) * (tmp_displacement_v != 0)
    return tmp_displacement_u, tmp_displacement_v, tmp_strain_exx, tmp_strain_exy, tmp_strain_eyy, mask_dice


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



def inverse_interpolation(u=None, v=None, load_name=None, save_name='_Fine', case_id = 1065, load=True, blue_size=35):
    tmp_def_img = cv2.imread(os.path.join(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain', str(case_id)+'_d.bmp'), cv2.IMREAD_GRAYSCALE)
    img_size = tmp_def_img.shape
    if load:
        tmp_displacement_v, tmp_displacement_u, _, _, _, _ = load_ncorr_mat_result(fname=os.path.join(ncorr_results_path, 'CASE_'+str(case_id)+load_name+'.mat'))
    else:
        tmp_displacement_u, tmp_displacement_v = u, v

    my_disp_filter = tf.GaussianBlur(kernel_size=(blue_size, blue_size), sigma=blue_size//2+1)
    my_interpolator = interpolation.interpolator()
    mask_disp = torch.from_numpy(tmp_displacement_u != 0.0).to(torch.float32)

    cat_uv = torch.stack([torch.from_numpy(tmp_displacement_u), torch.from_numpy(tmp_displacement_v), torch.ones_like(mask_disp)], dim=0)
    cat_uv = my_disp_filter(cat_uv)
    u, v = cat_uv[0], cat_uv[1]

    np.save(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain\\' + str(case_id)+save_name+'.npy', np.stack([u.numpy(), v.numpy()], axis=0))

    grid_x, grid_y = torch.meshgrid(torch.arange(img_size[0]), torch.arange(img_size[1]))

    u_pos = grid_x + u
    v_pos = grid_y + v

    tmp_def_img = torch.from_numpy(tmp_def_img)
    img_inter = my_interpolator.interpolation(u_pos, v_pos, tmp_def_img, img_mode=True)
    cv2.imwrite(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain\\'+str(case_id)+save_name+'.bmp', img_inter.numpy().astype('uint8'))


def deformation_composition(fine_name='', raw_name='', init_name='', case_id=1065, ret=False, device='cuda', prev_radius=10, mask=None):
    tmp_def_img = cv2.imread(os.path.join(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain', str(case_id)+'_d.bmp'), cv2.IMREAD_GRAYSCALE)
    img_size = tmp_def_img.shape

    raw_displacemnt = np.load(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain\\' + str(case_id)+raw_name+'.npy').astype('float32')
    raw_v, raw_u = torch.from_numpy(raw_displacemnt[0]).to(device), torch.from_numpy(raw_displacemnt[1]).to(device)

    grid_x, grid_y = torch.meshgrid(torch.arange(img_size[1]), torch.arange(img_size[0]))
    my_interpolator = interpolation.interpolator()
    my_strain_filter = DifferNet_FIT_Inter(device=device, radius=7, skip=1)

    u_label, v_label, exx_label, exy_label, eyy_label, mask_label = load_label(str(case_id) + '_img&disp.npy')
    if init_name!= '':
        tmp_init_u, tmp_init_v, tmp_init_exx, tmp_init_exy, tmp_init_eyy, mask_init = load_ncorr_mat_result(fname=os.path.join(ncorr_results_path, 'CASE_' + str(case_id)+init_name+ '.mat'))
        mae_disp_dice, epe_disp_dice = error_summary(labels=[u_label, v_label], target=[tmp_init_u, tmp_init_v], mask=mask, )  # savemat_dir=os.path.join(save_dir, str(case_id)+'_uv_ncorr.mat')
        mae_strain_dice, epe_strain_dice = error_summary(labels=[exx_label, exy_label, eyy_label],
                                                         target=[tmp_init_exx, tmp_init_exy, tmp_init_eyy],
                                                         mask=mask, )  # savemat_dir=os.path.join(save_dir, str(case_id)+'_uv_ncorr.mat')

    tmp_fine_u, tmp_fine_v, _, _, _, _ = load_ncorr_mat_result(fname=os.path.join(ncorr_results_path, 'CASE_' + str(case_id) + fine_name + '.mat'))

    u_pos = grid_x + torch.from_numpy(tmp_fine_v)
    v_pos = grid_y + torch.from_numpy(tmp_fine_u)
    u_pos = u_pos.to(device)
    v_pos = v_pos.to(device)
    if mask is None:
        mask = ((tmp_fine_u != 0) + (tmp_fine_v != 0)) > 0

    [u_raw_inter, v_raw_inter] = my_interpolator.interpolation_list(
        u_pos=u_pos,
        v_pos=v_pos,
        gray_array_list=[raw_u, raw_v],
        img_mode=False,
        kernel='bspline'
    )
    u_final = u_raw_inter.cpu().numpy() + tmp_fine_u
    v_final = v_raw_inter.cpu().numpy() + tmp_fine_v
    if ret:
        return u_final, v_final

    strain_final = my_strain_filter(torch.stack([torch.from_numpy(u_final), torch.from_numpy(v_final)], dim=0).unsqueeze(0), with_net=False)
    strain = strain_final.detach().cpu().numpy()
    u_x, u_y, v_x, v_y = strain[0, 0], strain[0, 1], strain[0, 2], strain[0, 3]
    tmp_final_exx = u_x + 0.5 * (u_x ** 2 + v_x ** 2)
    tmp_final_exy = 0.5 * (u_y + v_x + u_x * u_y + v_x * v_y)
    tmp_final_eyy = v_y + 0.5 * (u_y ** 2 + v_y ** 2)

    u_fine = torch.stack([torch.from_numpy(tmp_fine_u), torch.from_numpy(tmp_fine_v)], dim=0).unsqueeze(0)
    u_raw_bi = torch.stack([raw_u, raw_v], dim=0).unsqueeze(0)
    d_u_fine__d_x_0 = my_strain_filter(u_fine, with_net=False)
    d_u_raw__d_x_raw = my_strain_filter(u_raw_bi, with_net=False)
    strain_decomp = (1 + d_u_raw__d_x_raw) * d_u_fine__d_x_0 + d_u_raw__d_x_raw
    strain = strain_decomp.detach().cpu().numpy()
    u_x, u_y, v_x, v_y = strain[0, 0], strain[0, 1], strain[0, 2], strain[0, 3]
    tmp_decomp_exx = u_x + 0.5 * (u_x ** 2 + v_x ** 2)
    tmp_decomp_exy = 0.5 * (u_y + v_x + u_x * u_y + v_x * v_y)
    tmp_decomp_eyy = v_y + 0.5 * (u_y ** 2 + v_y ** 2)

    mask_error = tmp_decomp_exx < (1.2 * np.max(mask * exx_label))
    mask = (mask * mask_error)
    mask_error = tmp_decomp_exx > (1.2 * np.min(mask * exx_label))
    mask = (mask * mask_error)
    mask_error = tmp_decomp_exy < (1.2 * np.max(mask * exy_label))
    mask = (mask * mask_error)
    mask_error = tmp_decomp_exy > (1.2 * np.min(mask * exy_label))
    mask = (mask * mask_error)
    mask_error = tmp_decomp_eyy < (1.2 * np.max(mask * eyy_label))
    mask = (mask * mask_error)
    mask_error = tmp_decomp_eyy > (1.2 * np.min(mask * eyy_label))
    mask = (mask * mask_error)
    #        (np.abs(mask * tmp_decomp_exx) < (1.2 * np.max(mask * exx_label))) * (np.abs(mask * tmp_decomp_exx) > (1.2 * np.min(mask * exx_label))) *\
    #        (np.abs(mask * tmp_decomp_exy) < (1.2 * np.max(mask * exy_label))) * (np.abs(mask * tmp_decomp_exy) > (1.2 * np.min(mask * exy_label)))*\
    #        (np.abs(mask * tmp_decomp_eyy) < (1.2 * np.max(mask * eyy_label))) * (np.abs(mask * tmp_decomp_eyy) > (1.2 * np.min(mask * eyy_label)))
    print('Raw: ', raw_name, ' Fine: ', fine_name)
    mae_disp_net, epe_disp_net = error_summary(labels=[u_label, v_label], target=[u_final, v_final], mask=mask, )  # savemat_dir=os.path.join(save_dir, str(case_id)+'_uv_net.mat')
    mae_strain_dice, epe_strain_dice = error_summary(labels=[exx_label, exy_label, eyy_label], target=[tmp_final_exx, tmp_final_exy, tmp_final_eyy], mask=mask, )  # savemat_dir=os.path.join(save_dir, str(case_id)+'_uv_ncorr.mat')
    # mae_disp_net, epe_disp_net = error_summary(labels=[exx_label, exy_label, eyy_label], target=[tmp_decomp_exx, tmp_decomp_exy, tmp_decomp_eyy],  mask=mask, )  # savemat_dir=os.path.join(save_dir, str(case_id)+'_uv_net.mat')

    show_img(mat_list=[[u_label, v_label], [u_final, v_final], [u_final-u_label, v_final-v_label]], mask=mask)

    save_ue = u_final-u_label
    save_ve = v_final-v_label
    save_ue[mask==0] = np.nan
    save_ve[mask==0] = np.nan
    save_exx = tmp_final_exx-exx_label
    save_exy = tmp_final_exy-exy_label
    save_eyy = tmp_final_eyy-eyy_label
    save_exx[mask==0] = np.nan
    save_exy[mask==0] = np.nan
    save_eyy[mask==0] = np.nan
    tmp_final_exx[mask==0] = np.nan
    tmp_final_exy[mask==0] = np.nan
    tmp_final_eyy[mask==0] = np.nan
    tmp_fine_u[mask==0] = np.nan
    tmp_fine_v[mask==0] = np.nan
    savemat(os.path.join(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain\R25Results', fine_name+'.mat'), {'ue': save_ue, 'ufine':tmp_fine_u,
                                                                                                                               've': save_ve, 'vfine':tmp_fine_v,
                                                                                                                               'exxe': save_exx, 'exx':tmp_final_exx,
                                                                                                                               'exye': save_exy, 'exy':tmp_final_exy,
                                                                                                                               'eyye': save_eyy, 'eyy':tmp_final_eyy})

    if init_name!= '':
        save_ue = u_final-tmp_init_u
        save_ve = v_final-tmp_init_v
        save_exx = tmp_init_exx-exx_label
        save_exy = tmp_init_exy-exy_label
        save_eyy = tmp_init_eyy-eyy_label
        save_exx[mask==0] = np.nan
        save_exy[mask==0] = np.nan
        save_eyy[mask==0] = np.nan
        tmp_init_exx[mask==0] = np.nan
        tmp_init_exy[mask==0] = np.nan
        tmp_init_eyy[mask==0] = np.nan
        save_ue[mask == 0] = np.nan
        save_ve[mask == 0] = np.nan
        u_final[mask==0] = np.nan
        v_final[mask==0] = np.nan
        savemat(os.path.join(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain\R25Results', init_name+'.mat'), {'ue': save_ue, 'ufine':u_final,
                                                                                                                                   've': save_ve, 'vfine': v_final,
                                                                                                                                   'exxe': save_exx, 'exx': tmp_final_exx,
                                                                                                                                   'exye': save_exy, 'exy': tmp_final_exy,
                                                                                                                                   'eyye': save_eyy, 'eyy': tmp_final_eyy})
    #
    # show_img(mat_list=[[exx_label, exy_label, eyy_label], [tmp_init_exx, tmp_init_exy, tmp_init_eyy], [tmp_final_exx, tmp_final_exy, tmp_final_eyy], [tmp_decomp_exx, tmp_decomp_exy, tmp_decomp_eyy]], mask=mask)


def eval_different_radius(case_id=1065, disp_radius=25, strain_radius=7, mask=None):

    displacemnt_fname = r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain\\CASE_' + str(case_id)+'_'+str(disp_radius)+'_'+str(strain_radius)+'.mat'

    u_label, v_label, exx_label, exy_label, eyy_label, mask_label = load_label(str(case_id) + '_img&disp.npy')
    tmp_init_u, tmp_init_v, tmp_init_exx, tmp_init_exy, tmp_init_eyy, mask_init = load_ncorr_mat_result(fname=displacemnt_fname)

    print('Disp Radius %d, Strain Radius %d'%(disp_radius, strain_radius))

    mae_disp_dice, epe_disp_dice = error_summary(labels=[u_label, v_label], target=[tmp_init_u, tmp_init_v], mask=mask, )  # savemat_dir=os.path.join(save_dir, str(case_id)+'_uv_ncorr.mat')
    mae_strain_dice, epe_strain_dice = error_summary(labels=[exx_label, exy_label, eyy_label], target=[tmp_init_exx, tmp_init_exy, tmp_init_eyy],  mask=mask, )  # savemat_dir=os.path.join(save_dir, str(case_id)+'_uv_ncorr.mat')


def eval_all_ncorr_1065():
    disp_radius_list =   [7, 10, 13, 16, 19, 22, 25, 28, 31]
    # strain_radius_list = [5, 5,  7,  7,  7,  7,  11, 7,  5]
    strain_radius_list = [7, 7,  7,  7,  7,  7,  7, 7,  7]
    mask = np.load(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain\mask.npy')
    for i in range(len(disp_radius_list)):
        eval_different_radius(disp_radius=disp_radius_list[i], strain_radius=strain_radius_list[i], mask=mask)



if __name__ == '__main__':
    '''1065'''
    mask = np.load(r'E:\Data\DATASET\LargeDeformationDIC\202403EVAL\ncorr_Decomp_strain\mask.npy')
    # inverse_interpolation(blue_size=9, case_id=1065, u=None, v=None, save_name='_31', load_name='_31_7', load=True)
    # inverse_interpolation(blue_size=9, case_id=1065, u=None, v=None, save_name='_10', load_name='_10_7', load=True)
    # deformation_composition(case_id=1065, raw_name='_10', fine_name='_10_F10', init_name='_10_7', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_25', fine_name='_25_F25', init_name='_10_7', ret=False, mask=mask)
    # inverse_interpolation(u=v_final, v=u_final, name='_Fine_2', load=False)

    # u, v = deformation_composition(case_id=1065, raw_name='_25', fine_name='_25_F25', init_name='_10_7', ret=True, mask=mask)
    # inverse_interpolation(blue_size=9, case_id=1065, u=v, v=u, save_name='_25_F25', load_name='', load=False)

    # deformation_composition(case_id=1065, raw_name='_10', fine_name='_10_F10', init_name='', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_10', fine_name='_10_F25', init_name='', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_10', fine_name='_10_F35', init_name='', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_25', fine_name='_25_F10', init_name='', ret=False, mask=mask)
    deformation_composition(case_id=1065, raw_name='_25', fine_name='_25_F25', init_name='_25_7', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_25', fine_name='_25_F35', init_name='', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_25_F25', fine_name='_25_F25_F10', init_name='', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_25_F25', fine_name='_25_F25_F25', init_name='', ret=False, mask=mask)
    deformation_composition(case_id=1065, raw_name='_25_F25', fine_name='_25_F25_F35', init_name='', ret=False, mask=mask)
    deformation_composition(case_id=1065, raw_name='_25_F25_F35', fine_name='_25_F25_F35_F37', init_name='', ret=False, mask=mask)
    deformation_composition(case_id=1065, raw_name='_25_F25_F35_F37', fine_name='_25_F25_F35_F37_F39', init_name='', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_25_F25_F25', fine_name='_25_F25_F25_F31', init_name='', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_31', fine_name='_31_F35', init_name='', ret=False, mask=mask)
    # deformation_composition(case_id=1065, raw_name='_31', fine_name='_31_F35_F35', init_name='', ret=False, mask=mask)
    # u, v = deformation_composition(case_id=1065, raw_name='_31_F35', fine_name='_31_F35_F35', init_name='', ret=True, mask=mask)
    # inverse_interpolation(blue_size=9, case_id=1065, u=v, v=u, save_name='_31_F35_F35', load=False)

    # u, v = deformation_composition(case_id=1065, raw_name='_25_F25', fine_name='_25_F25_F25', init_name='', ret=True, mask=mask)
    # u, v = deformation_composition(case_id=1065, raw_name='_25_F25_F35', fine_name='_25_F25_F35_F37', init_name='', ret=True, mask=mask)
    # inverse_interpolation(blue_size=9, case_id=1065, u=v, v=u, save_name='_25_F25_F35_F37', load=False)




    # 1st iter
    # deformation_composition(case_id=1065, fine_name='_25_F25', ret=False)
    # # 2nd iter
    # # u_final, v_final = deformation_composition(fine_name='_Fine', ret=True)
    # deformation_composition(case_id=1065, fine_name='_Fine_1', ret=False)
    # # u_final, v_final = deformation_composition(fine_name='_Fine_1', ret=True)
    # deformation_composition(case_id=1065, fine_name='_Fine_2', ret=False)

    '''Star-1'''
    # u_final, v_final = deformation_composition(fine_name='_Fine', ret=True)
    # deformation_composition(case_id=1001, fine_name='_Fine', ret=False)


    # inverse_interpolation(blue_size=9, case_id=1001, u=None, v=None, name='_Fine', load=True)
    # inverse_interpolation(u=v_final, v=u_final, name='_Fine_2', load=False)
    # inverse_interpolation(u=v_final, v=u_final, name='_INTER_2.bmp', load=False, blue_size=17)


    # eval_all_ncorr_1065()