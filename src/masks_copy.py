import cv2
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import os
import scipy
import random
import skimage.morphology as morphology
from pathlib import Path
import itertools
from shapely.geometry import Polygon
from skimage.draw import line
from all import *


def detect_contour(img):
    cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont_image = np.zeros_like(img)
    cv2.drawContours(cont_image, cont, -1, 255, 0)
    return cont, cont_image


def read_semantic_masks():
    folder_path = input("Введите путь к папке c семантическими масками: ")
    if os.path.exists(folder_path):
        parent_directory = os.path.dirname(folder_path)
    else:
        print(f"Путь  не существует.")

    new_folder_name = input("Введите имя новой папки для снимков сигнала: ")
    signal_path = os.path.join(parent_directory, new_folder_name)
    if Path(signal_path).exists:
        signal_path = signal_path
    else:
        try:
            os.mkdir(signal_path)
            print(f"Папка '{new_folder_name}' успешно создана по пути '{signal_path}'.")
        except OSError as error:
            print(f"Ошибка при создании папки: {error}")

    new_folder_name = input("Введите имя новой папки для шумных снимков: ")
    raw_path = os.path.join(parent_directory, new_folder_name)
    if Path(raw_path).exists:
        raw_path = raw_path
    else:
        try:
            os.mkdir(raw_path)
            print(f"Папка '{new_folder_name}' успешно создана по пути '{raw_path}'.")
        except OSError as error:
            print(f"Ошибка при создании папки: {error}")
    
    filenames_masks = os.listdir(folder_path)

    return filenames_masks, signal_path, raw_path, folder_path


def compute_previous_pixel(first_pixel, last_pixel, distance=1):
    x1, y1 = first_pixel
    x2, y2 = last_pixel

    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    length = 1 if length == 0 else length
    t = -distance / length
    x_t = np.round((x1 + t * (x2 - x1)), 0).astype(np.int32)
    y_t = np.round((y1 + t * (y2 - y1)), 0).astype(np.int32)

    return (x_t, y_t)

def compute_next_pixel(first_point, last_point, distance=1):
    x1, y1 = first_point
    x2, y2 = last_point

    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    t = 1 + distance / length
    x_t = np.round((x1 + t * (x2 - x1)), 0).astype(np.int32)
    y_t = np.round((y1 + t * (y2 - y1)), 0).astype(np.int32)

    return (x_t, y_t)


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


def closest_point(point, array):
     diff = array - point
     distance = np.einsum('ij,ij->i', diff, diff)
     return np.argmin(distance), distance


def check_valid_placement(existing_figures, new_figure_coordinates):
    for new_coord in new_figure_coordinates:
        x, y = new_coord

        # Проверка на пересечение или касание других фигур
        for existing_figure in existing_figures:
            if (x, y) in existing_figure:
                return False  # Фигура пересекается или касается другой

        # Проверка расстояния от соседних фигур
        for existing_figure in existing_figures:
            for existing_coord in existing_figure:
                ex, ey = existing_coord
                if abs(x - ex) <= 1 and abs(y - ey) <= 1:
                    return False  # Фигура находится на расстоянии <= 2 пикселей от соседних фигур

    return True


def distance(existing_figures, new_figure_coordinates):
    for new_coord in new_figure_coordinates:
        # for existing_figure in existing_figures:
        #     if new_coord in existing_figure:
        #         return -1
            
        for existing_figure in existing_figures:
            # for existing_coord in existing_figure:
            min_dist = float('inf')
            index, dist = closest_point(new_coord, existing_figure)
            if dist[index] < min_dist :
                min_dist = dist[index].item()
                nearest_point = existing_figure[index] 
                prev = [compute_previous_pixel(new_coord, nearest_point)]
                discrete_line = list(zip(*line(*prev[0], *nearest_point)))
                dist_ = len(discrete_line) - 1
                return dist_
            
        for existing_figure in existing_figures:
            for existing_coord in existing_figure:
                ex, ey = existing_coord
                if abs(new_coord[0] - ex) <= 1 and abs(new_coord[1] - ey) <= 1:
                    return 0




def make_sem_from_bin(folder_path, semantic_path):
    filenames_masks = os.listdir(folder_path)

   
    for file in filenames_masks[3:4]:
        bin_mask = cv2.imread(os.path.join(folder_path, file), 0)
        image_test = np.zeros_like(bin_mask)
        bin_mask = edit_bin_mask(bin_mask)
        cont, cont_image = detect_contour(np.clip(bin_mask, 0, 1)) # обнаружила все контура
        cont_array = [] # массив для всех точек всех новых контуров
        existing_polygons = []

        for contour in cont:
            
            intersect = True
            flag = True
            offset = 20

            while flag:
                # if temp == 3:
                #     temp = 4
                # offset = random.randint(3, temp-1) # выбор первочначальной ширины контура
                offset = random.randint(3, offset) # выбор первочначальной ширины контура

                print('offset now ', offset)
                temp = np.zeros_like(bin_mask) # создание временной картинки
                cv2.drawContours(temp, [contour], -1, 128, offset) # рисование контура с первоначальным offset
                c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
                temp = np.zeros_like(bin_mask) # опять создание временной картинки
                cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
                nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра

                nonzero = np.array([list(reversed(nonz)) for nonz in nonzero]) # создание массива со всеми точками
                if len(cont_array) == 0: 
                    cont_array.append(nonzero)
                    existing_polygons.append(Polygon(nonzero))
                    intersect = False
                    flag = False
                else:
                    dist = distance(cont_array, nonzero)
                    if dist == -1:
                        if offset > 3:
                            offset = offset - 1
                        else:
                            offset = 3
                            intersect = False
                    else:
                        cont_array.append(nonzero)
                        # if offset == of:
                        if dist == 0:
                            offset = 3

                        if dist < 20 and dist > 3:
                            print('dist < 20 = ', dist)
                            offset = dist//2
                        elif dist == 3:
                            offset = 3
                        elif dist > 20:
                            print('dist > 20, offset', offset)
                            if offset > 6:
                                offset = offset - 3
                            else:
                                offset = offset
                        intersect = False
                        

                        print('distance:', dist)


                    # if check_valid_placement(cont_array, nonzero):
                    # # if flag:
                    #     print('valid placement')
                    #     cont_array.append(nonzero)
                    #     intersect = False
                    #     # if offset == of:
                    #     offset = offset
                    #     # else:
                    #     #     offset = of
                    #     # cv2.drawContours(cont_image, [contour], -1, 128, offset)
                    #     # offset = 20
                    # else:
                    #     print('!!!!!!!!!!!!!! offset:', offset)
                    #     if offset > 3:
                    #         offset = offset - 1
                    #     else:
                    #         offset = 3
                    #         intersect = False
                            # draw new
                temp = np.zeros_like(bin_mask) # создание временной картинки
                cv2.drawContours(temp, [contour], -1, 128, offset) # рисование контура с первоначальным offset
                c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
                temp = np.zeros_like(bin_mask) # опять создание временной картинки
                cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
                nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра
                # plt.imshow(temp)
                nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
                cont_array.append(nonzero)

                cv2.drawContours(image_test, [contour], -1, 128, offset)
                cv2.drawContours(image_test, [contour], -1, 255, 0)
                ext, int, img = detect_contours(image_test)
                if len(ext) == len(int):
                    flag = False
                else:
                    cv2.drawContours(image_test, [contour], -1, 0, offset)
                    cv2.drawContours(image_test, [contour], -1, 0, 0)

                    


                        

            cv2.drawContours(cont_image, [contour], -1, 128, offset)
        cv2.drawContours(cont_image, cont, -1, 255, -1)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(cont_image, cv2.MORPH_CLOSE, kernel)

        cont_image = cv2.erode(cont_image, kernel, iterations=1)
        plt.imshow(cont_image)
        cv2.imwrite(os.path.join(semantic_path, file), cont_image.astype(np.uint8))


def make_sem_from_bin_another(folder_path, semantic_path):
    filenames_masks = os.listdir(folder_path)

   
    for file in filenames_masks[2:3]:
        print(file)
        bin_mask = cv2.imread(os.path.join(folder_path, file), 0)
        image_test = np.zeros_like(bin_mask)
        bin_mask = edit_bin_mask(bin_mask)
        cont, cont_image = detect_contour(np.clip(bin_mask, 0, 1)) # обнаружила все контура
        cont_array = [] # массив для всех точек всех новых контуров
        existing_polygons = []

        for contour in cont:
            
            intersect = True
            flag = True
            offset = 20

            while flag:
                print('len of cont array: ', len(cont_array))
                if len(cont_array) == 0:
                    offset = 3
                    temp = np.zeros_like(bin_mask) # создание временной картинки
                    cv2.drawContours(temp, [contour], -1, 128, offset) # рисование контура с первоначальным offset
                    c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
                    temp = np.zeros_like(bin_mask) # опять создание временной картинки
                    cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
                    nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра
                    nonzero = np.array([list(reversed(nonz)) for nonz in nonzero]) # создание массива со всеми точками
                    
                    cont_array.append(nonzero)
                    flag = False
                else:
                    temp = np.zeros_like(bin_mask) # создание временной картинки
                    cv2.drawContours(temp, [contour], -1, 128, 0) # рисование контура с первоначальным offset
                    # plt.imshow(temp)
                    # plt.show()
                    c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
                    temp = np.zeros_like(bin_mask) # опять создание временной картинки
                    cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
                    nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра

                    nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
                    dist = distance(cont_array, nonzero)
                    print('dist:', dist)
                    # if dist == -1:
                    #     offset = 3
                    #     print('Intersect even without offset, dist = ', dist)
                    if dist > 40:
                        offset = random.randint(4 , 20) - 1
                        print('dist > 40', dist)
                    elif dist < 40 and dist > 20:
                        offset = random.randint(3, 9)
                        print('dist < 40 and dist > 20', dist)
                    elif dist < 20 and dist > 3:
                        print('dist < 20 and dist > 3', dist)
                        offset = dist // 2 - 1
                        if offset < 3:
                            offset = 3

                    temp = np.zeros_like(bin_mask) # создание временной картинки
                    cv2.drawContours(temp, [contour], -1, 128, offset) # рисование контура с первоначальным offset
                    # plt.imshow(temp)
                    # plt.show()
                    c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
                    temp = np.zeros_like(bin_mask) # опять создание временной картинки
                    cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
                    nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра

                    nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
                    cont_array.append(nonzero)
                    
                    flag = False


                # if temp == 3:
                #     temp = 4
                # offset = random.randint(3, temp-1) # выбор первочначальной ширины контура

                # offset = random.randint(3, offset) # выбор первочначальной ширины контура

                # print('offset now ', offset)
                # temp = np.zeros_like(bin_mask) # создание временной картинки
                # cv2.drawContours(temp, [contour], -1, 128, offset) # рисование контура с первоначальным offset
                # c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
                # temp = np.zeros_like(bin_mask) # опять создание временной картинки
                # cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
                # nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра

                # nonzero = np.array([list(reversed(nonz)) for nonz in nonzero]) # создание массива со всеми точками
                # if len(cont_array) == 0: 
                #     cont_array.append(nonzero)
                #     existing_polygons.append(Polygon(nonzero))
                #     intersect = False
                #     flag = False
                # else:
                #     dist = distance(cont_array, nonzero)
                #     if dist == -1:
                #         if offset > 3:
                #             offset = offset - 1
                #         else:
                #             offset = 3
                #             intersect = False
                #     else:
                #         cont_array.append(nonzero)
                #         # if offset == of:
                #         if dist < 20 and dist > 3:
                #             print('dist < 20 = ', dist)
                #             offset = dist//2
                #         elif dist == 3:
                #             offset = 3
                #         elif dist > 20:
                #             print('dist > 20, offset', offset)
                #             if offset > 6:
                #                 offset = offset - 3
                #             else:
                #                 offset = offset
                #         intersect = False
                        

                #         print('distance:', dist)


                #     # if check_valid_placement(cont_array, nonzero):
                #     # # if flag:
                #     #     print('valid placement')
                #     #     cont_array.append(nonzero)
                #     #     intersect = False
                #     #     # if offset == of:
                #     #     offset = offset
                #     #     # else:
                #     #     #     offset = of
                #     #     # cv2.drawContours(cont_image, [contour], -1, 128, offset)
                #     #     # offset = 20
                #     # else:
                #     #     print('!!!!!!!!!!!!!! offset:', offset)
                #     #     if offset > 3:
                #     #         offset = offset - 1
                #     #     else:
                #     #         offset = 3
                #     #         intersect = False
                #             # draw new
                # temp = np.zeros_like(bin_mask) # создание временной картинки
                # cv2.drawContours(temp, [contour], -1, 128, offset) # рисование контура с первоначальным offset
                # c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
                # temp = np.zeros_like(bin_mask) # опять создание временной картинки
                # cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
                # nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра
                # # plt.imshow(temp)
                # nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
                # cont_array.append(nonzero)

                # cv2.drawContours(image_test, [contour], -1, 128, offset)
                # cv2.drawContours(image_test, [contour], -1, 255, 0)
                # ext, int, img = detect_contours(image_test)
                # if len(ext) == len(int):
                #     flag = False
                # else:
                #     cv2.drawContours(image_test, [contour], -1, 0, offset)
                #     cv2.drawContours(image_test, [contour], -1, 0, 0)

                    


                        

            cv2.drawContours(cont_image, [contour], -1, 128, offset)
        cv2.drawContours(cont_image, cont, -1, 255, -1)
        # kernel = np.ones((3, 3), np.uint8)
        # closed = cv2.morphologyEx(cont_image, cv2.MORPH_CLOSE, kernel)

        # cont_image = cv2.erode(cont_image, kernel, iterations=1)
        plt.imshow(cont_image)
        cv2.imwrite(os.path.join(semantic_path, file), cont_image.astype(np.uint8))

# new new main
if __name__ == '__main__':
    print('main')
    # folder_path = input("Введите путь к папке c бинарными масками: ")
    folder_path = './data/bin_masks'
    parent_directory = os.path.dirname(folder_path)
    semantic_path = os.path.join(parent_directory, 'sem')
    os.makedirs(semantic_path, exist_ok=True)

    filenames_masks = os.listdir(folder_path)
    print(filenames_masks)
    make_sem_from_bin_another(folder_path, semantic_path)
