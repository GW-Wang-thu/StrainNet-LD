import numpy as np
import cv2
import matplotlib.pyplot as plt

import utils.interpolation as interpolation


class deformation():
    def __init__(self, imsize):
        self.imsize = imsize
        self.basic_v, self.basic_u = np.meshgrid(np.arange(0, imsize[1]), np.arange(0, imsize[0]))

    def get_random_displacement(self, amp=10):
        self.basic_v, self.basic_u = np.meshgrid(np.arange(0, self.imsize[1]), np.arange(0, self.imsize[0]))
        _, (u_1, v_1) = self.continuum_deformation(amp, wavelength=np.random.randint(50, 200))
        _, (u_2, v_2) = self.continuum_deformation(amp, wavelength=np.random.randint(40, 100))
        (u_pos, v_pos), (u_3, v_3) = self.crack_close_deformation()
        _, (u_4, v_4) = self.affine_deformation(amp)
        u_prev = u_1 + u_4 + u_3
        v_prev = v_1 + v_3 + v_4
        (_, _, def_mask), (u_2, v_2) = self.crack_open_deformation(u_prev, v_prev)
        pos_u = self.basic_u + u_1 + u_4 + u_3 + u_2
        pos_v = self.basic_v + v_1 + v_4 + v_3 + v_2
        mask = np.ones_like(self.basic_u)
        mask[pos_u < 0] = 0
        mask[pos_u >= self.imsize[0]] = 0
        mask[pos_v < 0] = 0
        mask[pos_v >= self.imsize[1]] = 0
        mask = mask * (1 - def_mask)

        return (pos_u, pos_v), (u_1 + u_2 + u_3 + u_4, v_1 + v_2 + v_3 + v_4), mask

    def get_random_displacement_no_crack(self, amp=10):
        minsize = min(self.imsize[0], self.imsize[1])
        if amp < 2:
            _, (u_1, v_1) = self.continuum_deformation(amp * 0.6, wavelength=np.random.randint(50, 150))
            _, (u_2, v_2) = self.continuum_deformation(amp * 0.8, wavelength=np.random.randint(minsize // 5, minsize // 3))
            _, (u_3, v_3) = self.affine_deformation(amp * 0.5)
        else:
            _, (u_1, v_1) = self.continuum_deformation(amp * 0.25, wavelength=np.random.randint(100, 300))
            _, (u_2, v_2) = self.continuum_deformation(amp * 1, wavelength=np.random.randint(minsize // 3, minsize // 2))
            _, (u_3, v_3) = self.affine_deformation(amp)
        mask = np.ones_like(self.basic_u)
        pos_u = self.basic_u + u_1 + u_2 + u_3
        pos_v = self.basic_v + v_1 + v_2 + v_3
        mask[pos_u < 0] = 0
        mask[pos_u >= self.imsize[0]] = 0
        mask[pos_v < 0] = 0
        mask[pos_v >= self.imsize[1]] = 0
        return (pos_u, pos_v), (u_1 + u_2 + u_3, v_1 + v_2 + v_3), mask

    def continuum_deformation(self, amp_r, wavelength):
        amp = np.random.randint(2, 12) * amp_r / 8
        disp_seed_array_u = (np.random.random(size=(self.imsize[0]//wavelength, self.imsize[1]//wavelength)) - 0.5) * amp * 2
        disp_seed_array_v = (np.random.random(size=(self.imsize[0]//wavelength, self.imsize[1]//wavelength)) - 0.5) * amp * 2
        disp_u = cv2.resize(disp_seed_array_u, dsize=(self.imsize[1], self.imsize[0]), interpolation=cv2.INTER_CUBIC)
        disp_v = cv2.resize(disp_seed_array_v, dsize=(self.imsize[1], self.imsize[0]), interpolation=cv2.INTER_CUBIC)
        pos_u = self.basic_u + disp_u
        pos_v = self.basic_v + disp_v
        return (pos_u, pos_v), (disp_u, disp_v)

    def crack_close_deformation(self):
        xc = np.random.randint(100, self.imsize[0]-100)+0.5
        yc = np.random.randint(100, self.imsize[1]-100)+0.5
        # theta_0 = np.random.randint(0, 55)
        # theta_1 = np.random.randint(theta_0 + 6, min(62, theta_0+10))
        theta_0 = np.random.randint(0, 55)
        theta_1 = np.random.randint(theta_0 + 5, min(60, theta_0+10))
        theta_0 = theta_0 / 10
        theta_1 = theta_1 / 10
        R = np.random.randint(150, self.imsize[0] // 1.5)
        if R > 400:
            amp = np.random.randint(8, 12) * R * (theta_1 - theta_0) / 200
        else:
            amp = np.random.randint(8, 12) * R * (theta_1 - theta_0) / 150
        print(amp)

        rand_1 = np.random.randint(2, 10) / 10
        rand_2 = np.random.randint(2, 8)
        rand_3 = np.random.randint(2, 10) / 10
        rand_4 = np.random.randint(2, 8)
        rand_5 = np.random.randint(2, 10) / 10
        theta_array = np.arctan((self.basic_v - yc) / (self.basic_u - xc)) * ((self.basic_u - xc) > 0) + (np.arctan((self.basic_v - yc) / (self.basic_u - xc)) + np.pi) * ((self.basic_u - xc) < 0)
        rho_array = np.sqrt((self.basic_v - yc)**2 + (self.basic_u - xc)**2)
        R = R + rand_1 * amp * np.sin(rand_2 * theta_array) + rand_3 * amp * np.cos(rand_4 * theta_array) + rand_5 * amp * (theta_array - theta_0) * (theta_array - theta_1)
        disp_mask = (theta_array > theta_0) * (theta_array < theta_1) * (rho_array < 2 * R) * (rho_array >0.5 * R)
        amp_array = amp * (theta_array - theta_0) * (theta_1 - theta_array) * np.sqrt(np.abs(rho_array-0.5*R)) * np.sqrt(np.abs(rho_array-2*R)) * 10 / R
        sgn_array = 2 * (rho_array > R) - 1
        direction_x = np.cos(theta_array)
        direction_y = np.sin(theta_array)
        disp_u = cv2.GaussianBlur(cv2.GaussianBlur(amp_array * direction_x * disp_mask * sgn_array, ksize=(5, 5), sigmaX=2), ksize=(5, 5), sigmaX=2)
        disp_v = cv2.GaussianBlur(cv2.GaussianBlur(amp_array * direction_y * disp_mask * sgn_array, ksize=(5, 5), sigmaX=2), ksize=(5, 5), sigmaX=2)
        u_pos = - disp_u + self.basic_u
        v_pos = - disp_v + self.basic_v
        zero_pos = (((u_pos - xc)**2 + (v_pos-yc)**2) > R **2) * (((self.basic_u-xc)**2 + (self.basic_v-yc)**2) < R**2)
        u_pos = u_pos + 1000000*zero_pos
        v_pos = v_pos + 1000000*zero_pos
        return (u_pos, v_pos), (-disp_u, -disp_v)

    def crack_open_deformation(self, u_prev, v_prev):
        xc = np.random.randint(100, self.imsize[0] - 100)+0.5
        yc = np.random.randint(100, self.imsize[1] - 100)+0.5
        # theta_0 = np.random.randint(0, 55)
        # theta_1 = np.random.randint(theta_0 + 6, min(62, theta_0+10))
        # R0 = np.random.randint(100, self.imsize[0] // 2)
        theta_0 = np.random.randint(0, 55)
        theta_1 = np.random.randint(theta_0 + 5, min(60, theta_0+10))
        theta_0 = theta_0 / 10
        theta_1 = theta_1 / 10
        R0 = np.random.randint(150, self.imsize[0] // 1.5)
        if R0 >400:
            amp = np.random.randint(8, 12) * R0 * (theta_1-theta_0) / 200
        else:
            amp = np.random.randint(8, 12) * R0 * (theta_1 - theta_0) / 150
        # amp = np.random.randint(8, 12) * R0 * (theta_1-theta_0) / 150
        print(amp)

        rand_1 = np.random.randint(2, 10) / 10
        rand_2 = np.random.randint(2, 8)
        rand_3 = np.random.randint(2, 10) / 10
        rand_4 = np.random.randint(2, 8)
        rand_5 = np.random.randint(2, 10) / 10
        theta_array = np.arctan((self.basic_v - yc) / (self.basic_u - xc)) * ((self.basic_u - xc) > 0) + (np.arctan((self.basic_v - yc) / (self.basic_u - xc)) + np.pi) * ((self.basic_u - xc) < 0)
        rho_array = np.sqrt((self.basic_v - yc)**2 + (self.basic_u - xc)**2)
        R = R0 + rand_1 * amp * np.sin(rand_2 * theta_array) + rand_3 * amp * np.cos(rand_4 * theta_array) + rand_5 * amp * (theta_array - theta_0) * (theta_array - theta_1)
        disp_mask = (theta_array > theta_0) * (theta_array < theta_1) * (rho_array < 2 * R) * (rho_array >0.5 * R)
        amp_array = amp * (theta_array - theta_0) * (theta_1 - theta_array) * np.sqrt(np.abs(rho_array-0.5*R)) * np.sqrt(np.abs(rho_array-2*R)) * 10 / R
        sgn_array = 2 * (rho_array > R) - 1
        direction_x = np.cos(theta_array)
        direction_y = np.sin(theta_array)
        disp_u = cv2.GaussianBlur(cv2.GaussianBlur(amp_array * direction_x * disp_mask * sgn_array, ksize=(5, 5), sigmaX=2), ksize=(5, 5), sigmaX=2)
        disp_v = cv2.GaussianBlur(cv2.GaussianBlur(amp_array * direction_y * disp_mask * sgn_array, ksize=(5, 5), sigmaX=2), ksize=(5, 5), sigmaX=2)
        u_pos = disp_u + self.basic_u
        v_pos = disp_v + self.basic_v
        zero_pos = (np.abs(np.sqrt((self.basic_u - u_prev - xc)**2 + (self.basic_v - v_prev - yc)**2) - R) < amp_array) * disp_mask
        return (u_pos, v_pos, zero_pos), (disp_u, disp_v)

    def affine_deformation(self, amp):
        rot_theta = np.random.randint(-1, 1) * amp / 500
        affine_ex = 1 + np.random.randint(-1, 1) * amp / 200
        affine_ey = 1 + np.random.randint(-1, 1) * amp / 200
        affine_shear_gx = np.random.randint(-1, 1) * amp / 200
        affine_shear_gy = np.random.randint(-1, 1) * amp / 200
        Tx = np.random.randint(-10, 10) * amp / 2
        Ty = np.random.randint(-10, 10) * amp / 5
        cent_u = (self.basic_u - self.imsize[0]//2)
        cent_v = (self.basic_v - self.imsize[1]//2)
        u = np.cos(rot_theta) * cent_u - np.sin(rot_theta) * cent_v
        v = np.sin(rot_theta) * cent_u + np.cos(rot_theta) * cent_v
        u = affine_ex * u
        v = affine_ey * v
        u = u + affine_shear_gx * v
        v = v + affine_shear_gy * u
        u += Tx
        v += Ty
        u_pos = u + self.imsize[0]//2
        v_pos = v + self.imsize[1]//2
        u = u_pos - self.basic_u
        v = v_pos - self.basic_v
        return (u_pos, v_pos), (u, v)

    def boundary_mask(self):    # used only in the loss function
        pass


if __name__ == '__main__':

    my_img = cv2.imread("../demoimg/Demoimg2.bmp", cv2.IMREAD_GRAYSCALE)
    imsize = my_img.shape
    my_displacements = deformation(imsize=imsize)
    my_interpolator = interpolation.interpolator(imsize=imsize)

    # # continuum_deformation
    # pos, disp = my_displacements.continuum_deformation(amp=10)
    # img = my_interpolator.interpolation(pos[0], pos[1], my_img)
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()
    #
    # affine_deformation
    pos, disp = my_displacements.affine_deformation()
    img = my_interpolator.interpolation(pos[0], pos[1], my_img)
    plt.imshow(img)
    plt.colorbar()
    plt.show()

    # # crack_close_deformation
    # pos, disp = my_displacements.crack_close_deformation(xc=100.5, yc=200.5, theta_0=0.2, theta_1=1.5, R=100, amp=10)
    # img = my_interpolator.interpolation(pos[0], pos[1], my_img)
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()
    #
    # # crack_open_deformation
    # (u_pos, v_pos, zero_pose), disp = my_displacements.crack_open_deformation(xc=100.5, yc=200.5, theta_0=0.2, theta_1=3, R0=200, amp=1)
    # img_ref = (1 - zero_pose) * my_img
    # img = my_interpolator.interpolation(u_pos, v_pos, img_ref)
    # plt.imshow(img_ref)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(disp[0])
    # plt.colorbar()
    # plt.show()