import cv2
import numpy as np
import matplotlib.pyplot as plt


def formula_second(img, angles, color_map, k, save_dir, file_name):
    signal = np.zeros_like(img, dtype=np.float32)
    alpha_bord = angles[img == 128]
    alpha_bord[alpha_bord==alpha_bord.min()] = np.radians(1)
    alpha_back = angles[img == 0]
    alpha_hole = angles[img == 255]
    # print(np.unique(alpha_hole), np.unique(alpha_back))
    # k = k * 
    signal[img == 0] = (k*(1/(np.abs(np.cos(np.radians(alpha_back + 1)))**(0.87)) - 1) + 1) * color_map[img==0]

    signal[img == 128] = (k * (1/(np.abs(np.cos(np.radians(90)-(np.radians(180 - 90) - alpha_bord)))**(0.87)) - 1) + 1) *color_map[img==128]
    signal[img == 255] = (k * (1 / (np.abs(np.cos(np.radians(alpha_hole + 1)))**(1.1)) - 1) + 1) * color_map[img==255]
    signal = np.clip(signal, 0, 255)
    # print(np.unique(signal))
    # signal = cv2.GaussianBlur(signal, (11,11), 0)
    signal = cv2.GaussianBlur(signal, (9,9), 0)
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    ax.imshow(signal)
    cv2.imwrite(f'{save_dir}/{file_name}', signal.astype(np.uint8))
    return signal