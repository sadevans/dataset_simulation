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

def make_sem_from_bin(folder_path, semantic_path):
    filenames_masks = os.listdir(folder_path)
    for file in filenames_masks:
        bin_mask = cv2.imread(os.path.join(folder_path, file), 0)
        bin_mask = edit_bin_mask(bin_mask)
        cont, cont_image = detect_contour(np.clip(bin_mask, 0, 1)) # обанружила все контура
        cont_array = [] # массив для всех точек всех новых контуров
        existing_polygons = []

        for contour in cont:
            print('new')
            intersect = True
            # temp_offset = 20
            offset = 20
            while intersect:
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
                plt.imshow(temp)
                nonzero = np.array([list(reversed(nonz)) for nonz in nonzero]) # создание массива со всеми точками
                # print(nonzero)
                # intersect = True
                if len(cont_array) == 0: 
                    cont_array.append(nonzero)
                    existing_polygons.append(Polygon(nonzero))
                    # print(existing_polygons)
                    intersect = False
                else:
                    # of, flag = check_valid_placement(cont_array, nonzero, offset)
                    if check_valid_placement(cont_array, nonzero):
                    # if flag:
                        print('valid placement')
                        cont_array.append(nonzero)
                        intersect = False
                        # if offset == of:
                        offset = offset
                        # else:
                        #     offset = of
                        # cv2.drawContours(cont_image, [contour], -1, 128, offset)
                        # offset = 20
                    else:
                        print('!!!!!!!!!!!!!! offset:', offset)
                        if offset > 3:
                            offset = offset - 1
                        else:
                            offset = 3
                            intersect = False

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

                            # cv2.drawContours(cont_image, [contour], -1, 128, offset)
                    
                    # new_polygon = Polygon(nonzero)
                    # k = 0
                    # for poly in existing_polygons:
                    #     if (new_polygon.buffer(1).distance(poly) <= 1 and poly.buffer(1).distance(new_polygon) <= 1) or\
                    #         new_polygon.intersects(poly):
                    #         # new_polygon.buffer(offset).distance(poly) <= offset and poly.buffer(offset).distance(new_polygon) <= offset:
                    #         print('!!!!!!!!')
                    #         k += 1
                    # if k == 0:
                    #     intersect = False

                    # while not intersect:
                    # for cont_ in cont_array:
                    #     intersect_ = np.array([x for x in set(tuple(x) for x in nonzero) & set(tuple(x) for x in cont_)])
                    #     distances = np.linalg.norm(cont_[:, np.newaxis, :] - nonzero, axis=2)

                    #     # index, dist = closest_point(point, cont_int)
                    #     # print(distances)
                    #     close_points = np.where(distances <= 1)
                    #     # print('close points', close_points[0])
                    #     # print(close_points[0])
                    #     if len(intersect_) == 0 and len(close_points[0]) == 0:
                    #         print('final offset')
                    #         intersect = False
                    #         offset = offset
                    #         temp = 20
                    #     else:
                    #         temp = offset
            cv2.drawContours(cont_image, [contour], -1, 128, offset)
        cv2.drawContours(cont_image, cont, -1, 255, -1)
        # kernel = np.ones((2, 2), np.uint8)
        # closed = cv2.morphologyEx(cont_image, cv2.MORPH_CLOSE, kernel)
        # cont_image = closed.copy()
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
    # print(folder_path, parent_directory, semantic_path)
    filenames_masks = os.listdir(folder_path)[0]
    print(filenames_masks)
    make_sem_from_bin(folder_path, semantic_path)
    # for file in filenames_masks:
    #     bin_mask = cv2.imread(os.path.join(folder_path, file), 0)
    #     bin_mask = edit_bin_mask(bin_mask)
    #     print(np.unique(bin_mask))
    #     cont, cont_image = detect_contour(np.clip(bin_mask, 0, 1)) # обанружила все контура
    #     cont_array = [] # массив для всех точек всех новых контуров
    #     for contour in cont:
    #         # print('im here')
    #         intersect = True
    #         # offset = 20
    #         while intersect:
    #             offset = random.randint(3, 20) # выбор первочначальной ширины контура
    #             # print('offset now ', offset)
    #             temp = np.zeros_like(bin_mask) # создание временной картинки
    #             cv2.drawContours(temp, [contour], -1, 128, offset) # рисование контура с первоначальным offset
    #             c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
    #             temp = np.zeros_like(bin_mask) # опять создание временной картинки
    #             cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
    #             nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра
    #             plt.imshow(temp)
    #             nonzero = np.array([list(reversed(nonz)) for nonz in nonzero]) # создание массива со всеми точками

    #             # intersect = True
    #             if len(cont_array) == 0: 
    #                 cont_array.append(nonzero)
    #                 intersect = False
    #             else:
    #                 # while not intersect:
    #                 for c in cont_array:
    #                     intersect_ = np.array([x for x in set(tuple(x) for x in nonzero) & set(tuple(x) for x in c)])
    #                     distances = np.linalg.norm(c[:, np.newaxis, :] - nonzero, axis=2)
    #                     # print(distances)
    #                     close_points = np.where(distances <= 1)
    #                     # print('close points', close_points[0])
    #                     # print(len(close_points[0]))
    #                     if len(intersect_) == 0 and len(close_points[0]) == 0:
    #                         # print('final offset')
    #                         intersect = False
    #                         offset = offset
    #         cv2.drawContours(cont_image, [contour], -1, 128, offset)
    #     cv2.drawContours(cont_image, cont, -1, 255, -1)
    #     plt.imshow(cont_image)
    #     cv2.imwrite(os.path.join(semantic_path, filenames_masks[0]), cont_image.astype(np.uint8))

    #     plt.show()
