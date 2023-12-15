import cv2
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import os
import scipy
import random
import skimage.morphology as morphology
from pathlib import Path


def detect_contour(img):
    cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    cont_image = np.zeros_like(img)
    cv2.drawContours(cont_image, cont, -1, 255, 0)
    plt.imshow(cont_image) 
    
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





if __name__ == '__main__':
    folder_path = input("Введите путь к папке c бинарными масками: ")

    if os.path.exists(folder_path):
        parent_directory = os.path.dirname(folder_path)
        new_folder_name = input("Введите имя новой папки для семантических масок: ")
        semantic_path = os.path.join(parent_directory, new_folder_name)

        try:
            os.mkdir(semantic_path)
            print(f"Папка '{new_folder_name}' успешно создана по пути '{semantic_path}'.")
        except OSError as error:
            print(f"Ошибка при создании папки: {error}")

        new_folder_name = input("Введите имя новой папки для шумных снимков: ")
        raw_path = os.path.join(parent_directory, new_folder_name)

        try:
            os.mkdir(raw_path)
            print(f"Папка '{new_folder_name}' успешно создана по пути '{raw_path}'.")
        except OSError as error:
            print(f"Ошибка при создании папки: {error}")

    else:
        print(f"Путь  не существует.")

    
    filenames_masks = os.listdir(folder_path)

    for file_name in filenames_masks:
        bin_mask = cv2.imread(os.path.join(folder_path, file_name), 0)
        
        cont, cont_image = detect_contour(bin_mask)
        for c in cont:
            offset = random.randint(3, 20)
            cv2.drawContours(cont_image, c, -1, 128, offset)
            
        # cv2.drawContours(cont_image, dst_np, -1, 128, -1)
        cv2.drawContours(cont_image, cont, -1, 255, -1)

        cv2.imwrite(os.path.join(semantic_path, file_name), cont_image.astype(np.uint8))

        plt.imshow(cont_image)

    