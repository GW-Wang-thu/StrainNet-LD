import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import torch

from displacement_generation import deformation
from utils.interpolation import interpolator


class img_loader():
    def __init__(self, imdir, seg_size=(512, 512)):
        files = os.listdir(imdir)
        self.seg_size = seg_size
        self.all_imgs = [os.path.join(imdir, file) for file in files if file.endswith(".bmp")]

    def load_sample(self, idx):

        temp_img = cv2.imread(self.all_imgs[idx], cv2.IMREAD_GRAYSCALE)
        imsize0 = temp_img.shape

        height = imsize0[0]
        width = imsize0[1]

        assert height > self.seg_size[0]
        assert width > self.seg_size[1]

        rand = np.random.rand()
        stp_x = np.random.randint(0, height - self.seg_size[0] - 1)
        stp_y = np.random.randint(0, width - self.seg_size[1] - 1)

        if rand < 0.6:
            return temp_img[stp_x:stp_x+self.seg_size[0], stp_y:stp_y+self.seg_size[1]].T
        else:
            return cv2.rotate(temp_img, cv2.ROTATE_90_CLOCKWISE)[stp_x:stp_x+self.seg_size[0], stp_y:stp_y+self.seg_size[1]].T

    def __len__(self):
        return len(self.all_imgs)


def generate_dataset(num=500, amp='normal'):
    raw_image_dir = r'E:\Data\DATASET\LargeDeformationDIC\raw_imgs'
    save_dir = r"E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\\"
    img_size = (896, 896)
    my_displacements = deformation(imsize=img_size)
    my_img_loader = img_loader(imdir=raw_image_dir, seg_size=img_size)
    my_interpolator = interpolator()
    if amp == 'small':
        kernel = np.array([[-0.25, 0, 0.25], [-0.5, 0, 0.5], [-0.25, 0, 0.25]]) * 0.5
        kernel_y = kernel.T
        save_dir = r"F:\DATASET\StrainNet_LD\Dataset_Tiny\\"
    for i in range(num):
        if os.path.exists(save_dir + str(i) + "_img_d.bmp"):
            continue
        print(my_img_loader.all_imgs[i%len(my_img_loader)])
        temp_img = my_img_loader.load_sample(idx=i%len(my_img_loader))
        if temp_img.shape != img_size:
            continue
        # if np.random.rand() < 0.7:
        #     if np.random.rand() < 0.7:
        #         pos, disp, def_mask = my_displacements.get_random_displacement(amp=np.random.randint(2, 10))
        #     else:
        #         pos, disp, def_mask = my_displacements.get_random_displacement(amp=np.random.randint(1, 11) / 6)
        # else:
        if amp == 'small':
            if np.random.rand() < 0.3:
                pos, disp, mask = my_displacements.get_random_displacement_no_crack(amp=np.random.randint(1, 12) / 20)
            elif np.random.rand() < 0.6:
                pos, disp, mask = my_displacements.get_random_displacement_no_crack(amp=np.random.randint(12, 20) / 30)
            else:
                pos, disp, mask = my_displacements.get_random_displacement_no_crack(amp=np.random.randint(20, 40) / 40)
            print(np.max(np.abs(disp)))
        else:
            if np.random.rand() < 0.2:
                pos, disp, mask = my_displacements.get_random_displacement_no_crack(amp=np.random.randint(2, 20) / 2)
            elif np.random.rand() < 0.6:
                pos, disp, mask = my_displacements.get_random_displacement_no_crack(amp=np.random.randint(20, 50) / 2)
            else:
                pos, disp, mask = my_displacements.get_random_displacement_no_crack(amp=np.random.randint(40, 100) / 2)
        try:
            temp_ref_img = my_interpolator.interpolation(torch.from_numpy(pos[0].astype("float32")), torch.from_numpy(pos[1].astype("float32")), torch.from_numpy(temp_img.astype("float32"))).numpy()
        except:
            continue
        temp_def_img = temp_img + np.random.normal(0, 5, size=temp_img.shape)
        temp_def_img = temp_def_img * (temp_def_img >= 0)
        temp_def_img = temp_def_img * (temp_def_img <= 255) + (temp_def_img > 255) * 255
        temp_u = disp[0]
        temp_v = disp[1]

        np.save(save_dir + str(i) + "_imgs.npy", np.array([temp_ref_img, temp_def_img]))
        np.save(save_dir + str(i) + "_disps.npy", np.array([temp_u, temp_v]))
        cv2.imwrite(save_dir + str(i) + "_img_r.bmp", temp_ref_img)
        cv2.imwrite(save_dir + str(i) + "_img_d.bmp", temp_def_img)
        plt.figure(figsize=(9, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(temp_ref_img)
        plt.subplot(2, 2, 2)
        plt.imshow(mask)
        plt.subplot(2, 2, 3)
        plt.imshow(temp_u)
        plt.colorbar()
        plt.savefig(save_dir + str(i) + "_demo.png")
        plt.close()


def swap_dataset(amp='normal'):
    save_dir = r"E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\\"
    data_dir = r"E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Normal\\"
    if amp == 'small':
        # save_dir = r"E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Small\\"
        # data_dir = r"E:\Data\DATASET\LargeDeformationDIC\202403\Dataset_Small\\"
        save_dir = r"F:\DATASET\StrainNet_LD\Dataset_Tiny\\"
        data_dir = r"F:\DATASET\StrainNet_LD\Dataset_Tiny\\"

    train_percent = 0.8
    train_id_list = []
    valid_id_list = []
    if os.path.exists(save_dir + "Train\\id_list.csv"):
        train_id_list = np.loadtxt(save_dir + "Train\\id_list.csv", dtype="int32").tolist()
    if os.path.exists(save_dir + "Valid\\id_list.csv"):
        valid_id_list = np.loadtxt(save_dir + "Valid\\id_list.csv", dtype="int32").tolist()
    all_data = os.listdir(data_dir)
    all_imgs = [os.path.join(data_dir, file) for file in all_data if file.endswith("_imgs.npy")]
    all_disps = [os.path.join(data_dir, file) for file in all_data if file.endswith("_disps.npy")]

    k = 0
    for i in range(len(all_imgs) + len(train_id_list) + len(valid_id_list)):
        if i in train_id_list or i in valid_id_list:
            continue
        if np.random.rand() < train_percent:
            temp_type = "Train"
            train_id_list.append(k)
        else:
            temp_type = "Valid"
            valid_id_list.append(k)
        temp_imgs = np.load(all_imgs[k])
        temp_disps = np.load(all_disps[k])
        np.save(save_dir + temp_type + "//" + str(i + 10000) + "_img&disp.npy", np.concatenate([temp_imgs, temp_disps], axis=0))
        k += 1
    np.savetxt(save_dir + "Train/id_list.csv", np.array(train_id_list))
    np.savetxt(save_dir + "Valid/id_list.csv", np.array(valid_id_list))


if __name__ == '__main__':
    np.random.seed(123)
    # generate_dataset(num=2000, amp='normal')
    # swap_dataset(amp='normal')
    generate_dataset(num=1000, amp='small')
    swap_dataset(amp='small')

