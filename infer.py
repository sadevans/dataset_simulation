import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.morphology as morphology
import random
import os
from pathlib import Path
from tqdm import tqdm
from figure.fig import *
from src.transforms import *

def transform_w_bezier_new(img, ext, int):
    color_back = 110
    color_hole = 85

    width_img = np.zeros_like(img, dtype=np.float32)
    new_angles = np.zeros_like(img, dtype=np.float32)
    color_map = np.zeros_like(img, dtype=np.float32)


    for cont_ext, cont_int in zip(ext, int):
        for point in cont_ext:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_int)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_int[index] 
                    prev = [compute_previous_pixel(point, nearest_point)]
                    discrete_line = list(zip(*line(*prev[0], *nearest_point))) # find all pixels from the line
                    dist_ = len(discrete_line) - 2

                if dist_ > 2:
                    new_line = np.zeros(dist_*12, dtype=np.float32)
                    x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, 100, 700)

                    x, colors = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, color_hole, color_back)
                    reshaped_y  = np.array(y).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    averages_y = np.max(reshaped_y, axis=1)

                    reshaped_colors  = np.array(colors).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    angles = np.arctan(np.abs(np.gradient(y)))
                    new_angl = angles[::12]
                    reshaped_angls  = np.array(angles).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    averages_angls = np.max(reshaped_angls, axis=1)
                    max_indices = np.argmax(reshaped_angls, axis=1)
                    averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]
                   
                    draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=2)
                    draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)
                elif dist_ == 2:
                    new_line = [0] * 2 * 12
                    x, y = bezier(new_line, np.linspace(0, 1, len(new_line)),0.0, 100, 700)
                    x, colors = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, color_hole, color_back)

                    angles = np.arctan(np.abs(np.gradient(y)))
                    reshaped_angls  = np.array(angles).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    averages_angls = np.max(reshaped_angls, axis=1)
                    max_indices = np.argmax(reshaped_angls, axis=1)
                    reshaped_colors  = np.array(colors).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]
                    averages_angls = [averages_angls[0], averages_angls[0], averages_angls[1], averages_angls[1]]
                    averages_colors = [averages_colors[0], averages_colors[0], averages_colors[1], averages_colors[1]]

                    draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=2)
                    draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)

                elif dist_ == 1:
                    y = [700-10]
                    averages_colors = [(color_back - 2)]
                    angles = [np.arctan(y[0]/(dist_*12))]
                    cv2.line(new_angles, point, nearest_point, np.deg2rad(89), 2)
                    cv2.line(color_map, point, nearest_point, (np.tan(np.deg2rad(89))*12 * 110)/700, 2)

                cv2.line(width_img, point, nearest_point, dist_, 3)


    for cont_ext, cont_int in zip(ext, int):
        for point in cont_int:
                min_dist = float('inf')
                index, dist = closest_point(point, cont_ext)
                if dist[index] < min_dist :
                    min_dist = dist[index].item()
                    nearest_point = cont_ext[index]
                    next = [compute_next_pixel(point, nearest_point)]
                    discrete_line = list(zip(*line(*point, *next[0]))) # find all pixels from the line
                    distance = np.sqrt((point[0] - next[0][0])**2 + (point[1] - next[0][1])**2)
                    dist_ = len(discrete_line) - 2

                if dist_ == 0:
                    dist_ = 1

                if dist_ > 2:
                    new_line = np.zeros(dist_*12, dtype=np.float32)
                    x, y = bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, 100, 700)
                    x, colors = bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, color_hole, color_back)
                    reshaped_y  = np.array(y).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    averages_y = np.max(reshaped_y, axis=1)
                    reshaped_colors  = np.array(colors).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    angles = np.arctan(np.abs(np.gradient(y)))
                    reshaped_angls  = np.array(angles).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    averages_angls = np.max(reshaped_angls, axis=1)
                    max_indices = np.argmax(reshaped_angls, axis=1)
                    averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]

                elif dist_ == 2:
                    new_line = [0] * 2 * 12
                    x, y = bezier(new_line, np.linspace(0, 1, len(new_line)),0.0, 100, 700)
                    x, colors = bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, color_hole, color_back)
                    angles = np.arctan(np.abs(np.gradient(y)))
                    reshaped_angls  = np.array(angles).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    averages_angls = np.max(reshaped_angls, axis=1)
                    max_indices = np.argmax(reshaped_angls, axis=1)
                    reshaped_colors  = np.array(colors).reshape(-1, 12)  # Разбиваем на подмассивы по 12 элементов
                    averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]
                    averages_angls = [averages_angls[0], averages_angls[0], averages_angls[1], averages_angls[1]]
                    averages_colors = [averages_colors[0], averages_colors[0], averages_colors[1], averages_colors[1]]

                    # draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=2)
                    # draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=1)

                elif dist_ == 1:
                    y = [700-10]
                    averages_colors = [(color_back - 2)]
                    angles = [np.arctan(y[0]/(dist_*12))]
                    cv2.line(new_angles, point, nearest_point, np.deg2rad(89), 2)
                    cv2.line(color_map, point, nearest_point, (np.tan(np.deg2rad(89))*12 * 110)/700, 2)
                
                if new_angles[point[1], point[0]]  == 0:
                    draw_gradient_line(new_angles, next[0], discrete_line[-1::-1], averages_angls[-1::-1], thickness=2)
                if color_map[point[1], point[0]]  == 0:
                    draw_gradient_line(color_map, next[0], discrete_line[-1::-1], averages_colors[-1::-1], thickness=2)

                cv2.line(width_img, point, nearest_point, dist_, 2)


    img_cp = img.copy()
    mask = img_cp != 128 
    width_img[mask] = 0
    new_angles[mask] = 0
    color_map[img == 0] = color_back
    color_map[img == 255] = color_hole

    zeros = np.where((color_map==0) & (img==128))
    if len(zeros[0]) > 0:
        tmp = np.zeros_like(img)
        cv2.drawContours(tmp, [ext[0]], -1, 255, 0)
        c = detect_cont(tmp)
        ext = np.argwhere(tmp > 0)
        ext = np.array([list(reversed(ex)) for ex in ext])
        
        for i in range(len(zeros[0])):
            point = (zeros[0][i], zeros[1][i])

            index_int, dist_int = closest_point(point, cont_int)
            index_ext, dist_ext = closest_point(point, ext)
            nearest_point_int = cont_int[index_int]
            nearest_point_ext = ext[index_ext]
            discrete_line_int = list(zip(*line(*point, *nearest_point_int)))
            discrete_line_ext = list(zip(*line(*point, *nearest_point_ext)))
            distance_int = len(discrete_line_int)
            distance_ext = len(discrete_line_ext)
            if distance_int < distance_ext:
                val = color_back - distance_int*np.tan(new_angles[point[0], point[1]])/(distance_ext + distance_int)
            else:
                val = color_hole + distance_ext*np.tan(new_angles[point[0], point[1]])/(distance_ext + distance_int)

            if val == 85.0:
                val = (110.0 + 85.0)/2 -random.randint(10, 14)
            elif val == 110.0:
                val = (110.0 + 85.0)/2 -random.randint(10, 14)
            color_map[point[0], point[1]] = np.abs(val)
    color_map[img == 0] = 0
    color_map[img == 255] = 0
    return width_img, new_angles, color_map


def formula_second1(img, angles, color_map, k):
    signal = np.zeros_like(img, dtype=np.float32)
    alpha_bord = angles[img == 128]
    alpha_bord[alpha_bord==alpha_bord.min()] = np.radians(1)
    alpha_back = angles[img == 0]
    alpha_hole = angles[img == 255]
    signal[img == 0] = (k*(1/(np.abs(np.cos(np.radians(alpha_back + 1)))**(0.87)) - 1) + 1) * color_map[img==0]

    signal[img == 128] = (k * (1/(np.abs(np.cos(np.radians(90)-(np.radians(180 - 90) - alpha_bord)))**(0.87)) - 1) + 1) *color_map[img==128]
    signal[img == 255] = (k * (1 / (np.abs(np.cos(np.radians(alpha_hole + 1)))**(1.1)) - 1) + 1) * color_map[img==255]
    signal = np.clip(signal, 0, 255)
    # signal = cv2.GaussianBlur(signal, (11,11), 0)
    # signal = cv2.GaussianBlur(signal, (9,9), 0)
    # cv2.imwrite(f'{save_dir}/{file_name}', signal.astype(np.uint8))
    return signal


def edit_bin_mask(mask):
    copy = mask.copy()
    mask_bool = mask >= 128
    mask[mask_bool] = 255
    mask[copy < 128] = 0

    kernel = np.ones((5, 5), np.uint8)  # Можно изменить размер ядра по желанию
    mask = ~mask

    opening = np.clip((morphology.remove_small_holes(mask, 10)), 0, 1) * 255
    result = opening.copy()
    result[opening == 0] = 255
    result[opening == 255] = 0
    return result.astype(np.uint8)


def detect_cont(img):
    cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cont

def touch(img, first_contours, objects, i):
    temp_touch = np.zeros_like(img)
    # cv2.fillPoly(can, [objects[i].border], 128)
    # if objects[i].border_width == 3:
    cv2.drawContours(temp_touch, [first_contours[i]], -1, 128, 2*(objects[i].border_width))
    cv2.drawContours(temp_touch, [first_contours[i]], -1, 128, -1)
    temp_touch_2 = np.zeros_like(img)
    cv2.drawContours(temp_touch_2, [objects[i].border], -1, 255, -1)
    cv2.drawContours(temp_touch_2, [first_contours[i]], -1, 128, 2*(objects[i].border_width))
    cv2.drawContours(temp_touch_2, [first_contours[i]], -1, 128, -1)

    arr = np.where((temp_touch == 0) & (temp_touch_2==255))
    if len(arr[0]) > 0:
        return True
    return False

if __name__ == '__main__':
    color_back = 110
    color_hole = 85
    flag_signal = True
    flag_sem_masks = True

    method = 'bezier'
    if method == 'bezier': # okay
        transform = np.vectorize(transform_w_bezier)
        # width, color_map, new_angles = transform_w_bezier(img, ext, int) # bezier
        k = 0.125

    elif method == 'parabola':
        transform = np.vectorize(transform_w_parabola)
        # width, angles_img, new_angles, color_map = transform_w_parabola(img, ext, int) # parabola
        k = 0.5
    elif method == 'parabola_radius':
        transform = np.vectorize(transform_radius)
        # width, new_angles, color_map = transform_radius(img, ext, int) # parabola radius
        k = 0.5
    a = 50
    b = 10

    # folder_path = '/home/sasha/WSLProjects/dataset_simulation/data/bin_masks'
    folder_path = '/home/sasha/WSLProjects/dataset_simulation/data/test_png_labels'

    # folder_path = save_semantic_dir
    parent_directory = Path(folder_path).parent
    signal_path = os.path.join(parent_directory, 'test_png_signal1')
    # raw_path = os.path.join(parent_directory, 'raw')
    raw_path = '/home/sasha/WSLProjects/dataset_simulation/data/test_png_raw1'

    # os.makedirs(signal_path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)

    # print(folder_path)
    parent_directory = os.path.dirname(folder_path)
    semantic_path = '/home/sasha/WSLProjects/dataset_simulation/data/test_png_labels_semantic1'
    # semantic_path = os.path.join(parent_directory, 'sem')
    os.makedirs(semantic_path, exist_ok=True)

    filenames_masks = os.listdir(folder_path)
    filenames_masks.sort()
    # print(filenames_masks)
    first_contours = []
    for file in tqdm(filenames_masks):
    # for file in tqdm(['20412.png']):
    # for file in tqdm(['20331.png']):
    # for file in tqdm(['20001.png']):
    # for file in tqdm(['20005.png']):
    # for file in tqdm(['20000.png']):
    # for file in tqdm(['20907.png']):
    # for file in tqdm(['20331.png']):
    # for file in tqdm(['circles.png']):
        flag_touching = True
        # print(file)

        objects = [] # массив 
        img = cv2.imread(os.path.join(folder_path, file), 0)
        img = edit_bin_mask(img)
        # first_contours.append(detect_cont(img))

        first_contours = detect_cont(img)

        # отрисовываем контуры, чтобы получить все точки, создаем объекты Figure
        for contour in first_contours:
            temp = np.zeros_like(img)                     # создание временной картинки
            cv2.drawContours(temp, [contour], -1, 255, -1) # рисование контура с первоначальным offset


            c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # детектирование контура

            obj = Figure(c[0].reshape(-1, 2)) # создаем объекты класса Figure
            objects.append(obj)

        # for i, obj in enumerate(objects):
        for i in range(len(objects)):
            # m = i + 2
            if len(objects[i].contour) > 1000:
                m = random.randint(3, 3)
            elif len(objects[i].contour) > 600:
                m = random.randint(3, 5)
            elif len(objects[i].contour) > 150:
                m = random.randint(3, 10)
            else:
                m = random.randint(3, 20)

            # flag_render = False
            break_flag = False
            for j in range(3, m + 1):
                # сначала пытаемся отрендерить с данной шириной
                # и проверяем, не пересекается ли с другими границами или объектами
                obj_num, flag_render = objects[i].try_render(img, objects, objects[i], i, j) # может быть еще что-то надо передавать
                # print('flag, bad obj num, width, max width: ', flag_render, obj_num, j, m)
                while flag_render==False:
                    if j-2 >= 3: 
                        objects[i].border_width -= 2
                        obj_num, flag_render = objects[i].try_render(img, objects, objects[i], i, objects[i].border_width)
                        # print('im here, obj, flag, width = ', obj_num, flag_render, objects[i].border_width)
                        break_flag = True
                    else:
                        if objects[obj_num].border_width - 2 >= 3:
                            objects[obj_num].border_width -= 2
                            objects[obj_num].init_border_contour(img, objects[obj_num])
                            obj_num, flag_render = objects[i].try_render(img, objects, objects[i], i, objects[i].border_width)
                            # print('NOW here, obj, flag = ', obj_num, flag_render)
                        else:
                            objects[i].border_width = 3
                            obj_num, flag_render = objects[i].try_render(img, objects, objects[i], i, objects[i].border_width)
                            # print('HERE, obj, flag, width = ', obj_num, flag_render, objects[i].border_width)
                            # break_flag = True
                            if obj_num is not None and objects[i].border_width == 3 and objects[obj_num].border_width == 3:
                                flag_touching = True
                            flag_render = True
                            break_flag = True

                if break_flag:
                    break

        res = np.zeros_like(img)

        for i in range(len(objects)):
            bord_img_0_x = any(0 in coord for coord in zip(objects[i].border[:,0]))
            bord_img_n_x = any(img.shape[1] in coord for coord in zip(objects[i].border[:,0]))
            # bord_img_0_x = any(0 in coord for coord in zip(*objects[i].border))
            bord_img_n_y = any(img.shape[0] in coord for coord in zip(objects[i].border[:,1]))
            bord_img_0_y = any(0 in coord for coord in zip(objects[i].border[:,1]))


            if bord_img_0_x or bord_img_n_x:
                objects[i] = None
                continue
            elif bord_img_0_y or bord_img_n_y:
                objects[i] = None
                continue
            else:
                # tm = np.zeros_like(img)
                # cv2.drawContours(tm , [objects[i].border], -1, 128, 0)
                # cv2.drawContours(tm , [objects[i].contour], -1, 255, 0)
                # plt.imshow(tm)
                # plt.show()
                # UNCOMMENT
                if objects[i].border_width > 3:
                    # print()
                    after_touch = False

                    # while objects[i].border_width > 1 and touch(img, first_contours, objects, i):
                    temp_touch = np.zeros_like(img)
                    # cv2.fillPoly(can, [objects[i].border], 128)
                    # if objects[i].border_width == 3:
                    cv2.drawContours(temp_touch, [first_contours[i]], -1, 128, 2*(objects[i].border_width))
                    cv2.drawContours(temp_touch, [first_contours[i]], -1, 128, -1)
                    temp_touch_2 = np.zeros_like(img)
                    cv2.drawContours(temp_touch_2, [objects[i].border], -1, 255, -1)
                    cv2.drawContours(temp_touch_2, [first_contours[i]], -1, 128, 2*(objects[i].border_width))
                    cv2.drawContours(temp_touch_2, [first_contours[i]], -1, 128, -1)
                    # fig, ax = plt.subplots(1, 2, figsize=(10,10))
                    # ax[0].imshow(temp_touch)
                    # ax[0].set_title('> 3 temp touch')
                    # ax[1].imshow(temp_touch_2)
                    # ax[1].set_title('> 3 temp touch 2')
                    # plt.show()

                    arr = np.where((temp_touch == 0) & (temp_touch_2==255))
                    print(arr[0])
                    if len(arr[0]) > 0:
                    # print('TOUCH')
                    # print('> 3 ', len(objects[i].contour), objects[i].border_width)
                        temp = np.zeros_like(img)
                        cv2.drawContours(temp, [first_contours[i]], -1, 128, 2*(objects[i].border_width-2))
                        temp_cont = detect_cont(temp)
                        objects[i].border = temp_cont[0].reshape(-1, 2).copy()
                        objects[i].border_width -= 2 
                    cv2.drawContours(res, [first_contours[i]], -1, 128, 2*(objects[i].border_width-1))
                    temp = np.zeros_like(img)
                    cv2.drawContours(temp, [first_contours[i]], -1, 128, 2*(objects[i].border_width))
                    zeros = np.where((temp == 255) & (res==0))
                    
                    if len(zeros[0]) > 0:
                        res[zeros[0][:], zeros[1][:]] = 128

                else:
                    if touch(img, first_contours, objects, i) and objects[i].border_width>1:
                        temp = np.zeros_like(img)
                        # cv2.drawContours(temp, [first_contours[i]], -1, 128, 2*(objects[i].border_width-1))
                        cv2.drawContours(temp, [first_contours[i]], -1, 128, 2*(1))

                        temp_cont = detect_cont(temp)
                        objects[i].border = temp_cont[0].reshape(-1, 2).copy()
                        objects[i].border_width = 1
                    cv2.drawContours(res, [first_contours[i]], -1, 128, 2*(objects[i].border_width))

                    temp = np.zeros_like(img)
                    cv2.drawContours(temp, [first_contours[i]], -1, 128, 2*(objects[i].border_width))
                    zeros = np.where((temp == 255) & (res==0))
                    if len(zeros[0]) > 0:
                        res[zeros[0][:], zeros[1][:]] = 128
                cv2.drawContours(res, [first_contours[i]], -1, 255, -1)
        cv2.imwrite(f'{semantic_path}/{file}', res.astype(np.uint8))


        # for each contour
        width_full = np.zeros_like(img)
        angles_full = np.zeros_like(img)
        colors_full = np.zeros_like(img)

        if flag_signal:
            signal_full = np.zeros_like(img)

            if flag_touching:
                for i in tqdm(range(len(objects))):
                    if objects[i] is not None:
                        temp = np.zeros_like(img)
                        cv2.drawContours(temp, [objects[i].border], -1, 128, -1)
                        cv2.drawContours(temp, [objects[i].contour], -1, 255, -1)

                        width, new_angles, color_map = transform_w_bezier_new(temp, [objects[i].border], [objects[i].contour])
                        color_map[temp == 0] = color_back
                        color_map[temp == 255] = color_hole
                        new_angles[temp != 128] = 0.0
                        signal = formula_second1(res, new_angles, color_map, k)
                        nonzero = np.argwhere(signal != 0)
                        for pixel in nonzero:
                            if temp[pixel[0], pixel[1]] != 0:
                                signal_full[pixel[0], pixel[1]] = signal[pixel[0], pixel[1]]

                            elif res[pixel[0], pixel[1]] == 0:
                                signal_full[pixel[0], pixel[1]] = signal[pixel[0], pixel[1]]

                signal_full[signal_full == 0] = np.unique(signal_full)[1] + 1
                signal_full = cv2.GaussianBlur(signal_full, (11,11), 0)

            else:
                ext, int = detect_contours(res)
                # print(len(int), len(ext))
                width, new_angles, color_map = transform_w_bezier_new(res, ext, int)
                color_map[res == 0] = color_back
                color_map[res == 255] = color_hole
                print(k)
                signal_full = formula_second1(res, new_angles, color_map, k)
                signal_full = cv2.GaussianBlur(signal_full, (11,11), 0)

            # cv2.imwrite(f'{signal_path}/{file[:-4]}_{method}_signal_test.png', signal_full.astype(np.uint8))
            cv2.imwrite(f'{signal_path}/{file}.png', signal_full.astype(np.uint8))
            a = random.randint(40, 50)
            b = random.randint(5, 10)
            # a = 70
            # b = 8
            m = Gamma(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))
            sample = m.sample()
            sample.item()

            clean = torch.Tensor(signal_full)
            noisy = clean +  m.sample()* torch.randn(clean.shape)
            res_noisy = clean + m.sample()* torch.randn(clean.shape)
            # cv2.imwrite(f'{raw_path}/{file[:-4]}' + '_' + method + '_raw.png', np.clip(res_noisy.numpy(), 0, 255).astype(np.uint8))
            cv2.imwrite(f'{raw_path}/{file}', np.clip(res_noisy.numpy(), 0, 255).astype(np.uint8))