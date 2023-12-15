import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from skimage.draw import line
import torch
from torch.distributions.gamma import Gamma

from src.all import *
from src.bezier import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def transform_w_bezier(img, ext, int):
    color_back = 110
    color_hole = 85

    width_img = np.zeros_like(img, dtype=np.float32)
    new_angles = np.zeros_like(img, dtype=np.float32)
    color_map = np.zeros_like(img, dtype=np.float32)
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole
    for cont_ext, cont_int in zip(ext, int):
        for point in cont_ext:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_int)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_int[index] 
                    prev = [compute_previous_pixel(point, nearest_point)]
                    # discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
                    discrete_line = list(zip(*line(*prev[0], *nearest_point))) # find all pixels from the line
                    dist_ = len(discrete_line) - 1
                    # dist_ = len(discrete_line)

                    # print(dist_)
                new_line = np.zeros(dist_*12, dtype=np.float32)
                x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, 100, 700)
                x, colors = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, color_hole, color_back)
                # new_y = y[::12]
                reshaped_y  = np.array(y).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                averages_y = np.max(reshaped_y, axis=1)

                reshaped_colors  = np.array(colors).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                angles = np.arctan(np.abs(np.gradient(y)))
                new_angl = angles[::12]
                reshaped_angls  = np.array(angles).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                averages_angls = np.max(reshaped_angls, axis=1)
                max_indices = np.argmax(reshaped_angls, axis=1)
                averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]
                print(averages_colors)
                if len(averages_angls) != dist_:
                    print('dist = {:.2f}, len angls = {:.2f}, len line = {:.2f}'.format(dist_, len(averages_angls), len(discrete_line)))
                
                draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=4)
                draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=2)
                cv2.line(width_img, point, nearest_point, dist_, 3)


    for cont_ext, cont_int in zip(ext, int):
        for point in cont_int:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_ext)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_ext[index]
                    # prev = [compute_previous_pixel(point, nearest_point)]
                    next = [compute_next_pixel(point, nearest_point)]
                    # if point[0] != nearest_point[0] and point[1] != nearest_point[1]:
                    discrete_line = list(zip(*line(*point, *next[0]))) # find all pixels from the line
                    # else:
                    # discrete_line = list(zip(*line(*point, *nearest_point)))
                    dist_ = len(discrete_line)
                    print(dist_)

                if dist_ == 0:
                    dist_ = 1

                new_line = np.zeros(dist_*12, dtype=np.float32)
                x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, 100, 700)
                x, colors = bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, color_hole, color_back)

                # new_y = y[::12]
                reshaped_y  = np.array(y).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                averages_y = np.max(reshaped_y, axis=1)

                reshaped_colors  = np.array(colors).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                angles = np.arctan(np.abs(np.gradient(y)))

                # new_angl = angles[::12]
                reshaped_angls  = np.array(angles).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                averages_angls = np.max(reshaped_angls, axis=1)
                max_indices = np.argmax(reshaped_angls, axis=1)
                averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]
                if len(averages_angls) != dist_:
                    print('dist = {:.2f}, len angls = {:.2f}, len line = {:.2f}'.format(dist_, len(averages_angls), len(discrete_line))) 
                if new_angles[point[1], point[0]]  == 0 or dist_>10:
                    if dist_>10:

                        draw_gradient_line(color_map, nearest_point, discrete_line[-1::-1], averages_colors[-1::-1], thickness=2)
                        draw_gradient_line(new_angles, nearest_point, discrete_line[-1::-1], averages_angls[-1::-1], thickness=2)

                    # else:    
                    draw_gradient_line(new_angles, nearest_point, discrete_line[-1::-1], averages_angls[-1::-1], thickness=2)

    img_cp = img.copy()
    mask = img_cp != 128 
    width_img[mask] = 0
    new_angles[mask] = 0
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole
    return width_img, new_angles, color_map


def transform_w_parabola(img, ext, int):
    # ext, int = ext.tolist(), int.tolist()
    color_back = 110
    color_hole = 85

    width_img = np.zeros_like(img, dtype=np.float32)
    angles_img = np.zeros_like(img, dtype=np.float32)
    new_angles = np.zeros_like(img, dtype=np.float32)
    color_map = np.zeros_like(img, dtype=np.float32)
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole

    for cont_ext, cont_int in zip(ext, int):
        for point in cont_ext:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_int)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_int[index] 
                    prev = [compute_previous_pixel(point, nearest_point)]
                    next = [compute_next_pixel(point, nearest_point)]
                    discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
                    dist_ = len(discrete_line)

                angle = np.arctan(700 / (dist_ * 12))
                if dist_ <= 10: cv2.line(width_img, point, nearest_point, dist_, 3)
                if dist_ <= 10:cv2.line(angles_img, point, nearest_point, angle, 2)

                if len(discrete_line) == 1:
                    height_vals = [color_back - 2]
                    discrete_line_ = discrete_line
                else:
                    height_vals = [((k_ * ((color_back - color_hole) / dist_)) + color_hole) for k_ in range(0, dist_+2)]
                    discrete_line_ = discrete_line + next

                draw_gradient_line(color_map, point, discrete_line_, height_vals[-1::-1], thickness=3)

                # calculate new angles
                if dist_ >= 3:
                        k = dist_/10
                        val = angle * 0.9

                        y_mean = val
                        y_0 = val/2
                        y_n = val/2
                        x_plot = np.arange(0, len(discrete_line))
                        coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                        a,b,c = coefs[0]
                        y_parabola = parabola(x_plot, a, b, c)
                        heights = y_parabola
                        draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=2)

    for cont_ext, cont_int in zip(ext, int):
        for point in cont_int:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_ext)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_ext[index]
                   
                    next = [compute_next_pixel(point, nearest_point)]
                    prev = [compute_previous_pixel(point, nearest_point)]

                    discrete_line = list(zip(*line(*point, *next[0]))) # find all pixels from the line
                    dist_ = len(discrete_line)

                if dist_ == 0:
                    dist_ = 1

                angle = np.arctan(700 / (dist_ * 12))
                cv2.line(width_img, point, nearest_point, dist_, 3)
                cv2.line(angles_img, point, nearest_point, angle, 2)

                if len(discrete_line) == 1:
                    height_vals = [color_back - 2]
                    discrete_line_ =discrete_line

                else:
                    height_vals = [((k_ * ((color_back - color_hole) / dist_)) + color_hole) for k_ in range(1, dist_+2)]
                    discrete_line_ = discrete_line + next

                if dist_ > 10:
                    if point[0] != nearest_point[0] and point[1] != nearest_point[1]:
                        draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)
  
                        val = angle * 0.9
                        y_mean = val
                        y_0 = val/2
                        y_n = val/2
                        x_plot = np.arange(0, len(discrete_line))
                        coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                        a,b,c = coefs[0]
                        y_parabola = parabola(x_plot, a, b, c)
                        heights = y_parabola
                        # print(y_parabola)
                        draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=2)
                # if dist_ <= 10:
                if new_angles[point[1], point[0]]  == 0 and dist_ > 10:
                    draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)
                     
                # calculate new angles
                # if dist_ >= 3 and dist_ <= 10:
                if dist_ >= 3 and new_angles[point[1], point[0]]  == 0:     
                     
                # if dist_ >= 3:
                        val = angle * 0.9
                        y_mean = val
                        y_0 = val/2
                        y_n = val/2
                        x_plot = np.arange(0, len(discrete_line))
                        coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                        a,b,c = coefs[0]
                        y_parabola = parabola(x_plot, a, b, c)
                        heights = y_parabola
                        # print(y_parabola)
                        draw_gradient_line(new_angles, nearest_point, discrete_line[-1::-1], np.abs(y_parabola[-1::-1]), thickness=2)
                   
    mask_width = width_img == 3

    if len(np.unique(mask_width))>1:
        val3 = np.max(new_angles[mask_width])
        new_angles[width_img == 2] = val3*1.001
        new_angles[width_img == 1] = val3*1.002
    else:
        new_angles[width_img == 2] = angles_img[width_img == 2] * 0.9
        new_angles[width_img == 1] = angles_img[width_img == 1] * 0.9

    mask = img != 128 
    width_img[mask] = 0
    angles_img[mask] = 0
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole
    new_angles[mask] = 0

    return width_img, new_angles, color_map


def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def transform_radius(img, ext, int):
    # ext, int = ext.tolist(), int.tolist()

    color_back = 110
    color_hole = 85

    width_img = img.copy()
    angles_img = img.copy()
    new_angles = np.zeros_like(img, dtype=np.float32)
    color_map = img.copy()
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole

    width_img[img == 0] = 0
    width_img[img == 255] = 0
    flag = True
    for cont_ext, cont_int in zip(ext, int):
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [cont_int], 2)
        img = np.zeros_like(img)
        cont = cont_ext.copy()
        for point in cont:
            radius = 1
            flag = True
            while flag:
                mask = np.zeros_like(img)
                mask = cv2.circle(mask, point, radius, 255, -1)
                mask = cv2.inRange(mask, 255,255)
                cont_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                mask_cont = np.zeros_like(img)
                cv2.drawContours(mask_cont, cont_mask,  -1, 1, 0)
                nonzero = np.argwhere(mask_cont > 0)
                nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
                intersect = np.array([x for x in set(tuple(x) for x in nonzero) & set(tuple(x) for x in cont_int)])
                if len(intersect) != 0:
                    flag=False
                else:
                    radius += 1

            dist = float('inf')
            for dot in intersect:
                dist_ = np.sqrt((point[0] - dot[0])**2 + (point[1] - dot[1])**2)
                if dist_ < dist:
                    dist = ((np.round(dist_, 0)).astype(np.int32)).item() + 1
                    nearest_point = dot
            
            discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
            # discrete_line_x = np.array(list(zip(*discrete_line))[0])
            
            mid_point = [(point[0] + nearest_point[0])//2, (point[1] + nearest_point[1])//2]
            prev = [compute_previous_pixel(point, nearest_point)]
            next = [compute_next_pixel(point, nearest_point)]


            angle = np.arctan(700 / (dist * 12))
            cv2.line(width_img, point, nearest_point, dist, 3)
            cv2.line(angles_img, point, nearest_point, angle, 2)

            if len(discrete_line) == 1:
                height_vals = [color_back - 2]
                discrete_line_ =discrete_line

            else:
                height_vals = [((k_ * ((color_back - color_hole) / dist)) + color_hole) for k_ in range(1, dist+2)]
                # discrete_line_ = discrete_line + next
                discrete_line_ = discrete_line

            draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)
            if dist_ >= 2:
                    val = angle * 0.9
                    y_mean = val
                    y_0 = val/2
                    y_n = val/2
                    x_plot = np.arange(0, len(discrete_line))
                    coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                    a,b,c = coefs[0]
                    y_parabola = parabola(x_plot, a, b, c)
                    heights = y_parabola
                    draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=1)

    for cont_ext, cont_int in zip(ext, int):
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [cont_int], 2)
        img = np.zeros_like(img)
        modes = ['internal', 'external']
        cont = cont_int.copy()
        for point in cont:
            radius = 1
            flag = True
            while flag:
                mask = np.zeros_like(img)
                mask = cv2.circle(mask, point, radius, 255, -1)
                mask = cv2.inRange(mask, 255,255)
                cont_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                mask_cont = np.zeros_like(img)
                cv2.drawContours(mask_cont, cont_mask,  -1, 1, 0)
                nonzero = np.argwhere(mask_cont > 0)
                nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
                
                intersect = np.array([x for x in set(tuple(x) for x in nonzero) & set(tuple(x) for x in cont_ext)])
                if len(intersect) != 0:
                    flag=False
                else:
                    radius += 1

            dist = float('inf')
            for dot in intersect:
                dist_ = np.sqrt((point[0] - dot[0])**2 + (point[1] - dot[1])**2)
                if dist_ < dist:
                    dist = ((np.round(dist_, 0)).astype(np.int32)).item() + 1
                    nearest_point = dot
            
            discrete_line = list(zip(*line(*point, *nearest_point))) # find all pixels from the line
            discrete_line_x = np.array(list(zip(*discrete_line))[0])
            
            mid_point = [(point[0] + nearest_point[0])//2, (point[1] + nearest_point[1])//2]
            prev = [compute_previous_pixel(point, nearest_point)]
            next = [compute_next_pixel(point, nearest_point)]


            angle = np.arctan(700 / (dist * 12))
            cv2.line(width_img, point, nearest_point, dist, 3)
            cv2.line(angles_img, point, nearest_point, angle, 2)

            if len(discrete_line) == 1:
                height_vals = [color_back - 2]
                discrete_line_ =discrete_line

            else:
                height_vals = [((k_ * ((color_back - color_hole) / dist)) + color_hole) for k_ in range(1, dist+2)]
                # discrete_line_ = discrete_line + next
                discrete_line_ = discrete_line
            if point[0] != nearest_point[0] and point[1] != nearest_point[1]:
                draw_gradient_line(color_map, point, discrete_line_, height_vals[-1::-1], thickness=3)

            else: draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)

            # if dist_ <= 10: draw_gradient_line(color_map, point, discrete_line_, height_vals, thickness=3)
            # calculate new angles
            # if dist_ >= 2 and dist_ <= 10:
            if dist_ >= 2:


                    val = angle * 0.9
                    y_mean = val
                    y_0 = val/2
                    y_n = val/2
                    x_plot = np.arange(0, len(discrete_line))
                    coefs = curve_fit(parabola, [x_plot[0], x_plot[len(x_plot)//2], x_plot[-1]], [y_0, y_mean, y_n])
                    a,b,c = coefs[0]
                    y_parabola = parabola(x_plot, a, b, c)
                    heights = y_parabola
                    draw_gradient_line(new_angles, point, discrete_line, np.abs(y_parabola), thickness=1)

    mask_width = width_img == 3
    # img = img.copy()

    mask = img != 128 
    width_img[mask] = 0
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole
    new_angles[mask] = 0
    return width_img, new_angles, color_map