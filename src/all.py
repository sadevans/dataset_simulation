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
from figure.fig import Figure

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# class Img():
#     def __init__(self, image):

def detect_cont(img):
    cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cont

def detect_contours(img):
    # objects = []
    # mask_hole = cv2.inRange(img, 255, 255)
    # mask_bord = cv2.inRange(img, 128, 255)

    # c_int_ = detect_cont(mask_hole)
    # c_ext_ = detect_cont(mask_bord)

    # set_borders = [set(map(tuple, np.argwhere(cv2.drawContours(np.zeros_like(img), [c_ext_[i]], -1, 128, -1)==128))) for i in range(len(c_ext_))]
    # set_holes = [set(map(tuple, np.argwhere(cv2.drawContours(np.zeros_like(img), [c_int_[j]], -1, 255, -1)==255))) for j in range(len(c_int_)))]

    # for i, set_border in enumerate(set_borders):
    #     for j, set_hole in enumerate(set_holes):
    #         if set_hole.issubset(set_border):
    #             obj = Figure(c_int_[j].reshape(-1, 2))
    #             obj.border = c_ext_[i].reshape(-1, 2).copy()
    #             objects.append(obj)
    objects = []
    mask_hole = cv2.inRange(img, 255, 255)
    mask_bord = cv2.inRange(img, 128, 255)

    c_int_ = detect_cont(mask_hole)
    c_ext_ = detect_cont(mask_bord)

    temp_ext = []
    temp_int = []

    for i in range(len(c_ext_)):
        for j in range(len(c_int_)):
            temp_bord = np.zeros_like(img)
            cv2.drawContours(temp_bord, [c_ext_[i]], -1, 128, -1)
            temp_hole = np.zeros_like(img)
            cv2.drawContours(temp_hole, [c_int_[j]], -1, 255, -1)
            set_border = set(map(tuple, np.argwhere(temp_bord==128)))
            set_hole = set(map(tuple, np.argwhere(temp_hole==255)))
            if set_hole.issubset(set_border):
                obj = Figure(c_int_[j].reshape(-1, 2)) # создаем объекты класса Figure
                obj.border = c_ext_[i].reshape(-1, 2).copy()
                objects.append(obj)

    # while len(c_ext) != np.minimum(len(c_ext_), len(c_int_)):
    #     temp_bord = np.zeros_like(img)
    #     cv2.drawContours(temp_bord, [c_ext_[i].reshape(-1, 2)], -1, 128, -1)
    #     temp_hole = np.zeros_like(img)
    #     cv2.drawContours(temp_hole, [c_int_[j].reshape(-1, 2)], -1, 255, -1)
    #     set_border = set(map(tuple, np.argwhere(temp_bord==128)))
    #     set_hole = set(map(tuple, np.argwhere(temp_hole==255)))
    #     if set_hole.issubset(set_border):
    #         print('SUBSET ', i, j)
    #         c_ext.append(c_ext_[i].reshape(-1, 2))
    #         c_int.append(c_int_[j].reshape(-1, 2))
    #         obj = Figure(int[i]) # создаем объекты класса Figure
    #         obj.border = ext[i].copy()
    #         objects.append(obj)

    #         if j == i:
    #             print('i == j: ', i, j)
    #             i+= 1
    #             # if j != len(c_int_):
    #             j += 1
    #         else:
    #             print('i != j: ', i, j)
    #             i+= 1
    #             j = temp_j
    #             print(i, j)

    #     else:
    #         print('NOT A SUBSET ', i, j)
    #         # if temp_j < j:
    #         temp_j = j
    #         print('TEMP J ',temp_j)

    #         if j != len(c_int_):
    #             j += 1
    

    # return c_ext, c_int
    return objects



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
