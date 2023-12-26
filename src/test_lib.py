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


def distance(img, existing_figures, new_figure_coordinates):
    temp = np.zeros_like(img)
    # fig, ax = plt.subplots(1,1, figsize=(30, 15))
    for new_coord in new_figure_coordinates:
        # for existing_figure in existing_figures:
        #     if new_coord in existing_figure:
        #         return -1
            
        
            
        # for existing_figure in existing_figures:
        #     for existing_coord in existing_figure:
        #         ex, ey = existing_coord
        #         if abs(new_coord[0] - ex) <= 1 and abs(new_coord[1] - ey) <= 1:
        #             return 0
                
        for existing_figure in existing_figures:
            print(existing_figure)
            print(new_coord)
            # for existing_coord in existing_figure:
            min_dist = float('inf')
            index, dist = closest_point(new_coord, existing_figure)
            if dist[index] < min_dist :
                min_dist = dist[index].item()
                nearest_point = existing_figure[index] 
                prev = [compute_previous_pixel(new_coord, nearest_point)]
                discrete_line = list(zip(*line(*prev[0], *nearest_point)))
                dist_ = len(discrete_line) - 1
                # dist_ = np.sqrt((new_coord[0] - nearest_point[0])**2 + (new_coord[1] - nearest_point[1])**2)
                min_dist = dist_
        
        return dist_


def detect_cont(img):
    cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cont_image = np.zeros_like(img)
    # cv2.drawContours(cont_image, cont, -1, 255, 0)
    # return cont, cont_image
    return cont


# def detect_boxes(cont):

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


def detect_boxes_contours(img):
    contours = detect_cont(img)
    
    arr = []
    contours_bin_array = []
    boxes_array = []
    offsets = []
    test_img = np.zeros_like(img)
    temporary_bin_image = np.zeros_like(img)
    for cont in contours:
        offset = 0
        temp = np.zeros_like(img) # создание временной картинки
        cv2.drawContours(temp, [cont], -1, 255, 0) # рисование контура с первоначальным offset
        c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
        temp = np.zeros_like(img) # опять создание временной картинки
        cv2.drawContours(temp, c, 0, 255, 0) # рисование большого контура
        nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра
        nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
        print('length of contour: ', len(nonzero))
        if len(contours_bin_array) == 0:
            print('first contour, offset = ', offset)
            contours_bin_array.append(nonzero)

            offset = 3
            offsets.append(offset)

            temp = np.zeros_like(img) # создание временной картинки
            cv2.drawContours(temp, [cont], -1, 128, offset) # рисование контура с первоначальным offset
            c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
            temp = np.zeros_like(img) # опять создание временной картинки
            cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
            nonzero_ = np.argwhere(temp > 0) # нахождение всех пикселей конутра
            nonzero_ = np.array([list(reversed(nonz)) for nonz in nonzero_]) # создание массива со всеми точками
            arr.append(nonzero_)

        else:
            # здесь смотреть расстояние
            offset = random.randint(3, 20)
            print('first choose offset: ', offset)
            dist = distance(img, arr, nonzero)
            print('computed distance: ', dist)
            # if dist == 0:
            #     contours_bin_array.append(nonzero)
            #     if offset > 3:
            #         offset = offset - 1
            #     else:
            #         offset = 3
            #         # intersect = False
            # else:
            #     contours_bin_array.append(nonzero)
            #     # if offset == of:
            #     # if dist == 0: # контур пересекся с каким-то
            #     #     print('Пересекаются')
            #     #     offset = 3

            #     if dist <= 2: # расстояние до ближайшего контура меньше двух пикселей
            #         # offset = offset - 2
            #         # offset = 3
            #         print('dist <= 2, offset = ', offset)

            #     elif dist < 20 and dist > 3: # расстояние приемлимое только для границы меньше 10 пикселей
            #         print('dist < 20 = ', dist)
            #         offset = random.randint(3, np.maximum(3, dist//2))
            #     elif dist == 3: # расстояние равно 3 пикселям
            #         offset = 3
            #     elif dist > 20: # расстояние больше 20 пикселей
            #         print('dist > 20, offset', offset)
            #         # if offset > 6:
            #         offset = random.randint(3, np.minimum(dist//2, 20))

            #     elif dist > 40:
            #         print('dist > 40, offset =  ', offset)
            #         offset = offset

            # offsets.append(offset)
            # temp = np.zeros_like(img) # создание временной картинки
            # cv2.drawContours(temp, [cont], -1, 128, offset) # рисование контура с первоначальным offset
            # c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
            # temp = np.zeros_like(img) # опять создание временной картинки
            # cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
            # nonzero_ = np.argwhere(temp > 0) # нахождение всех пикселей конутра
            # nonzero_ = np.array([list(reversed(nonz)) for nonz in nonzero_]) # создание массива со всеми точками
            # arr.append(nonzero_)
        

        # contours_bin_array.append(nonzero)
        x, y, w, h = cv2.boundingRect(nonzero)
        point1 = (x, y)         # original bounding box
        point2 = (x + w, y + h) # original bounding box

        print('offset after: ', offset)
        point1_offset = (x - offset, y - offset)                    # bounding box for offset 
        point2_offset = (point2[0] + offset, point2[1] + offset)    # bounding box for offset 

        print(x, y, w, h)
        # cv2.fillConvexPoly(test_img, nonzero, 255)
        # cv2.drawContours(test_img, )
        cv2.drawContours(test_img, [cont], -1, 128, offset)
        cv2.drawContours(test_img, [cont], -1, 255, -1)
        # cv2.rectangle(test_img, point1, point2, 120, 1)
        cv2.rectangle(test_img, point1_offset, point2_offset, 200, 1)


    plt.imshow(test_img)
    plt.show()


def detect_boxes_contours_new(img):
    contours = detect_cont(img)
    
    arr = []
    contours_bin_array = []
    boxes_array = []
    offsets = []
    test_img = np.zeros_like(img)
    temporary_bin_image = np.zeros_like(img)
    for cont in contours:
        offset = 0
        temp = np.zeros_like(img) # создание временной картинки
        cv2.drawContours(temp, [cont], -1, 255, 0) # рисование контура с первоначальным offset
        c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
        temp = np.zeros_like(img) # опять создание временной картинки
        cv2.drawContours(temp, c, 0, 255, 0) # рисование большого контура
        nonzero = np.argwhere(temp > 0) # нахождение всех пикселей конутра
        nonzero = np.array([list(reversed(nonz)) for nonz in nonzero])
        print('length of contour: ', len(nonzero))
        if len(contours_bin_array) == 0:
            print('first contour, offset = ', offset)
            contours_bin_array.append(nonzero)

            offset = 3 # рисуем первый контур с шириной границы 3
            offsets.append(offset)

            temp = np.zeros_like(img) # создание временной картинки
            cv2.drawContours(temp, [cont], -1, 128, offset) # рисование контура с первоначальным offset
            c, _ = cv2.findContours(np.clip(temp, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # детектирование контура
            temp = np.zeros_like(img) # опять создание временной картинки
            cv2.drawContours(temp, c, 0, 128, 0) # рисование большого контура
            nonzero_ = np.argwhere(temp > 0) # нахождение всех пикселей конутра
            nonzero_ = np.array([list(reversed(nonz)) for nonz in nonzero_]) # создание массива со всеми точками
            arr.append(nonzero_)

        else:
            # dist = 
            # здесь смотреть расстояние
            offset = random.randint(3, 20) # выбираем рандомный offset 
            print('first choose offset: ', offset)

            # dist = distance(img, arr, nonzero)
            # print('computed distance: ', dist)
            
        # contours_bin_array.append(nonzero)
        x, y, w, h = cv2.boundingRect(nonzero)
        point1 = (x, y)         # original bounding box
        point2 = (x + w, y + h) # original bounding box

        print('offset after: ', offset)
        point1_offset = (x - offset, y - offset)                    # bounding box for offset 
        point2_offset = (point2[0] + offset, point2[1] + offset)    # bounding box for offset 

        print(x, y, w, h)
        # cv2.fillConvexPoly(test_img, nonzero, 255)
        # cv2.drawContours(test_img, )
        cv2.drawContours(test_img, [cont], -1, 128, offset)
        cv2.drawContours(test_img, [cont], -1, 255, -1)
        # cv2.rectangle(test_img, point1, point2, 120, 1)
        cv2.rectangle(test_img, point1_offset, point2_offset, 200, 1)


    plt.imshow(test_img)
    plt.show()
    

if __name__ == '__main__':
    # folder_path = './'

    # parent_directory = Path(save_semantic_dir).parent
    # signal_path = os.path.join(parent_directory, 'signal')
    # raw_path = os.path.join(parent_directory, 'raw')
    # os.makedirs(signal_path, exist_ok=True)
    # os.makedirs(raw_path, exist_ok=True)

    folder_path = './data/bin_masks'
    parent_directory = os.path.dirname(folder_path)
    semantic_path = os.path.join(parent_directory, 'sem')
    os.makedirs(semantic_path, exist_ok=True)

    filenames_masks = os.listdir(folder_path)

    for file in filenames_masks[6:7]:
        print(file)
        img = cv2.imread(os.path.join(folder_path, file), 0)
        img = edit_bin_mask(img)

        detect_boxes_contours_new(img)


    