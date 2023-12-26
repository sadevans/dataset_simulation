import numpy as np

import cv2
import matplotlib.pyplot as plt

import scipy
from scipy.interpolate import CubicSpline, lagrange
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
import random
import os
from skimage.draw import line
from scipy.ndimage import gaussian_filter1d

import torch
from torch.distributions.gamma import Gamma

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)


def detect_contours(img):
    # kernel = np.ones((2, 2), np.uint8)
    # closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Применение операции вычитания для получения разницы между закрытым и исходным изображением
    # diff = cv2.absdiff(closed, img)

    # diff = cv2.erode(img, kernel, iterations=1)
    # diff = cv2.dilate(diff, kernel, iterations=1)

    # mask_bord = cv2.inRange(diff, 128, 128)
    mask_bord = cv2.inRange(img, 128,128)
    # plt.imshow(mask_bord)
    # plt.show()

    cont, hier = cv2.findContours(mask_bord, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # cont_small, hier_small = cv2.findContours(mask_bord, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mask_cont2 = np.zeros_like(img)
    mask_cont2 = cv2.drawContours(mask_cont2, cont, -1, 200, 0)
    # plt.imshow(mask_cont2)
    # plt.title('smalls')
    # plt.show()
    ext = []
    int = []
    test = np.zeros_like(img)
    mask_cont1 = np.zeros_like(img)
    k = 1
    # mask_cont = np.zeros_like(img)
    # cv2.drawContours(test, cont, -1, 255, -1)
    # plt.imshow(test)
    # plt.show()
    # print(cont[0])
    # print(hier)
    for i in range(len(cont)):
    # for i in range(len(hier[0])):
    # for i, j in zip(range(len(cont)), range(len(hier[0]))):
    # for i in range(2):
        print(cont[i])
        mask_cont = np.zeros_like(img)

        if hier[0][i][3] == -1:
            mask_cont = cv2.drawContours(mask_cont, [cont[i]], 0, 255, 0)
            c, _ = cv2.findContours(mask_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            nonz = cv2.bitwise_and(img, img, mask=mask_cont)
            # print(cont[i])
            # print(np.argwhere(nonz > 0))
            # plt.imshow(mask_cont)
            # plt.show()
            # cv2.fillPoly(mask_cont1, [cont[i]], 1)
            # nonzero = np.argwhere(mask_cont > 0)
            nonzero = np.argwhere(nonz > 0)

            nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
            print(type(cont[i]))
            print(len(cont[i]), len(nonzero), len(c[0]))
            # ext.append(nonzero)
            ext.append(cont[i].reshape(-1, 2))
            # print(nonzero)

            # k += 1
            # x, y, w, h = cv2.boundingRect(np.array(cont[i]))
            # # print(x, y)
            # if ((y != 0) and ((y + h) != mask_cont.shape[0])) or ((x != 0) and (x+w) != mask_cont.shape[1]):
            #     cv2.drawContours(test, [cont[i]], 0, 128, 0)
            #     nonzero = np.argwhere(mask_cont > 0)
            #     nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])

            #     ext.append(nonzero)
            # else:
            #     img[y:y+h, y:y+h, x:x+w] = 0

            # print(nonzero[127])


        else:
            
            mask_cont = cv2.drawContours(mask_cont, [cont[i]], 0, 255, 0)
            c, _ = cv2.findContours(mask_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            nonz = cv2.bitwise_and(img, img, mask=mask_cont)

            # nonzero = np.argwhere(mask_cont > 0)
            nonzero = np.argwhere(nonz > 0)

            nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
            print(type(cont[i]))

            print(len(cont[i]), len(nonzero), len(c[0]))
            int.append(cont[i].reshape(-1, 2))
            # if len(nonzero) > 20:
            #     int.append(nonzero)
            # print(nonzero)

            # plt.imshow(mask_cont)
            # plt.show()
            # x, y, w, h = cv2.boundingRect(np.array(cont[i]))
            # if ((y != 0) and ((y + h) != mask_cont.shape[0])) or ((x != 0) and (x+w) != mask_cont.shape[1]):
            #     cv2.drawContours(test, [cont[i]], 0, 255, 0)
                
            #     nonzero = np.argwhere(mask_cont > 0)
            #     nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
            #     int.append(nonzero)
            # else:
            #     img[y:y+h, x:x+w] = 0

    # plt.imshow(test)

    # plt.imshow(mask_cont2)
    # plt.imshow(test)
    # print('cont lens: ', len(ext), len(int))
    # cv2.drawContours(test, ext, -1, 128, 0)
    # cv2.drawContours(test, int, -1, 255, 0)


    # # plt.imshow(mask_cont)
    # plt.imshow(test)

    # plt.show()


    return ext, int


def closest_point(point, array):
    diff = array - point
    distance = np.einsum('ij,ij->i', diff, diff)
    return np.argmin(distance), distance


@torch.jit.script
def cuda_closest_point(point, array):
    diff = array - point.unsqueeze(0)
    distance = torch.sum(diff ** 2, dim=1)
    return torch.argmin(distance), distance


def parabola(x, a, b, c):
    return a*x**2 + b*x + c


def draw_gradient_line(img, start_point, points, colors, thickness=4, flag=None):
    start = start_point
    if flag is None:
        for i in range(1, len(points) - 1):
            # if img[start[1], start[0]] == 0 or img[start[1], start[0]] == 255 or img[start[1], start[0]]==128:
            if i+1 != len(points)-1:
                cv2.line(img, start, points[i+1], colors[i], thickness)
            start = points[i]
    elif flag == 'ext':
        for i in range(1, len(points) - 1):
            if img[start[1], start[0]] == 0 or img[start[1], start[0]] == 255 or img[start[1], start[0]]==128:
                print(start)
                if i+1 != len(points)-1:
                    cv2.line(img, start, points[i+1], colors[i], thickness)
                start = points[i]

# @torch.jit.script
def cuda_draw_gradient_line(img, start_point, points, colors, thickness=4, flag=None):
    start = start_point
    print(colors)
    colors = list(colors.cpu().detach().numpy())
    print(colors)
    print(points)

    print(flag)
    if flag is None:
        for i in range(1, len(points) - 1):
            if i+1 != len(points)-1:
                cv2.line(img.cpu().detach().numpy(), (start[0].item(), start[1].item()), (points[i+1][0].item(), points[i+1][1].item()), colors[i], thickness)
            start = points[i]
    elif flag == 'ext':
        print(points.size(0))
        print(len(points))
        for i in range(1, len(points) - 1):
            if img[start[1], start[0]] == 0 or img[start[1], start[0]] == 255 or img[start[1], start[0]]==128:
                print(start)
                if i+1 != len(points)-1:
                    cv2.line(img.cpu().detach().numpy(), (start[0].item(), start[1].item()), (points[i+1][0].item(), points[i+1][1].item()), colors[i].item(), thickness)
                start = points[i]


def compute_previous_pixel(first_pixel, last_pixel, distance=1):
    x1, y1 = first_pixel
    x2, y2 = last_pixel

    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    length = 1 if length == 0 else length
    t = -distance / length
    x_t = np.round((x1 + t * (x2 - x1)), 0).astype(np.int32)
    y_t = np.round((y1 + t * (y2 - y1)), 0).astype(np.int32)

    return (x_t, y_t)


# @torch.jit.script
def cuda_compute_previous_pixel(first_pixel, last_pixel, distance=1):
    diff = last_pixel - first_pixel
    length = torch.sqrt(torch.sum(diff ** 2))
    length = torch.where(length == 0, torch.tensor(1.0), length)
    t = -distance / length
    x_t = torch.round((first_pixel[0] + t * (last_pixel[0] - first_pixel[0]))).to(torch.int32)
    y_t = torch.round((first_pixel[1] + t * (last_pixel[1] - first_pixel[1]))).to(torch.int32)
    return x_t, y_t



def compute_next_pixel(first_point, last_point, distance=1):
    x1, y1 = first_point
    x2, y2 = last_point

    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    t = 1 + distance / length
    x_t = np.round((x1 + t * (x2 - x1)), 0).astype(np.int32)
    y_t = np.round((y1 + t * (y2 - y1)), 0).astype(np.int32)

    return (x_t, y_t)


# @torch.jit.script
def cuda_compute_next_pixel(first_point, last_point, distance=1):
    diff = last_point - first_point
    length = torch.sqrt(torch.sum(diff ** 2))
    t = 1 + distance / length
    x_t = torch.round((first_point[0] + t * (last_point[0] - first_point[0]))).to(torch.int32)
    y_t = torch.round((first_point[1] + t * (last_point[1] - first_point[1]))).to(torch.int32)
    return x_t, y_t
