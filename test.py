import numpy as np
import cv2
import src.all
from src.transforms import *
from src.masks import *
# from src.make_signal import *
from src.draw import *
import matplotlib.pyplot as plt
from pathlib import Path


# def formula_second1(img, angles, color_map, k, save_dir, file_name):
#     print(np.unique(img))
#     signal = np.zeros_like(img, dtype=np.float32)
#     alpha_bord = angles[img == 128]
#     # alpha_bord[alpha_bord==alpha_bord.min()] = np.radians(1)
#     alpha_back = angles[img == 0]
#     print(np.unique(alpha_back))
#     alpha_hole = angles[img == 255]
#     # print(np.unique(alpha_hole), np.unique(alpha_back))
#     # k = k * 
#     signal[img == 0] = (k*(1/(np.abs(np.cos(np.radians(alpha_back + 1)))**(0.87)) - 1) + 1) * color_map[img==0]
#     signal[img == 128] = (k * (1/(np.abs(np.cos(np.radians(90)-(np.radians(180 - 90) - alpha_bord)))**(0.87)) - 1) + 1) *color_map[img==128]
#     signal[img == 255] = (k * (1 / (np.abs(np.cos(np.radians(alpha_hole + 1)))**(1.1)) - 1) + 1) * color_map[img==255]
#     signal = np.clip(signal, 0, 255)
#     # print(np.unique(signal))
#     # signal = cv2.GaussianBlur(signal, (11,11), 0)
#     signal = cv2.GaussianBlur(signal, (9,9), 0)
#     print(np.unique(signal))
#     fig, ax = plt.subplots(1, 1, figsize=(30, 15))
#     ax.imshow(signal)
#     cv2.imwrite(f'{save_dir}/{file_name}', signal.astype(np.uint8))
#     return signal

def formula_second1(img, angles, color_map, k, save_dir, file_name):
    print(f'{save_dir}/{file_name}')
    signal = np.zeros_like(img, dtype=np.float32)
    alpha_bord = angles[img == 128]
    # alpha_bord[alpha_bord==alpha_bord.min()] = np.radians(1)
    alpha_back = angles[img == 0]
    alpha_hole = angles[img == 255]
    # print(np.unique(alpha_hole), np.unique(alpha_back))
    # k = k * 
    signal[img == 0] = (k*(1/(np.abs(np.cos(np.radians(alpha_back + 1)))**(0.87)) - 1) + 1) * color_map[img==0]

    signal[img == 128] = (k * (1/(np.abs(np.cos(np.radians(90)-(np.radians(180 - 90) - alpha_bord)))**(0.87)) - 1) + 1) *color_map[img==128]
    signal[img == 255] = (k * (1 / (np.abs(np.cos(np.radians(alpha_hole + 1)))**(1.1)) - 1) + 1) * color_map[img==255]
    signal = np.clip(signal, 0, 255)
    # signal = cv2.GaussianBlur(signal, (11,11), 0)
    signal = cv2.GaussianBlur(signal, (9,9), 0)
    cv2.imwrite(f'{save_dir}/{file_name}', signal.astype(np.uint8))
    return signal

if __name__ == '__main__':
    # method = input('Введите метод (bezier или parabola или parabola_radius): ')
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

    # for debug
    folder_path = '/home/sasha/WSLProjects/sem_simulator/simulate/data/semantic'
    parent_directory = os.path.dirname(folder_path)
    
    signal_path = os.path.join(parent_directory, 'signal')

    raw_path = os.path.join(parent_directory, 'raw')
    
    filenames = os.listdir(folder_path)

    # filenames, signal_path, raw_path, folder_path = read_semantic_masks()
    print(k)
    for file in filenames[3:]:
        img = cv2.imread(os.path.join(folder_path, file), 0)
        ext, int, img = detect_contours(img)
        width, new_angles, color_map = transform_w_bezier(img, ext, int) # okay +-
        # width, new_angles, color_map = transform_w_parabola(img, ext, int)

        signal = formula_second1(img, new_angles, color_map, k, signal_path, file[:-4]+ '_' + method + '_signal.png')

        m = Gamma(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))
        sample = m.sample()
        sample.item()

        clean = torch.Tensor(signal)
        noisy = clean +  m.sample()* torch.randn(clean.shape)
        res = clean + m.sample()* torch.randn(clean.shape)
        cv2.imwrite(f'{raw_path}/{file[:-4]}' + '_' + method + '_raw.png', np.clip(res.numpy(), 0, 255).astype(np.uint8))

    y = 100

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    ax[0,0].imshow(width)
    ax[0,0].set_title(f'width')

    ax[0,1].imshow(color_map)
    ax[0,1].set_title(f'color_map')

    ax[0,2].imshow(new_angles)
    ax[0,2].set_title(f'new angles')

    ax[1,1].plot(color_map[y, :])
    ax[1,1].grid()


    ax[1,0].plot(width[y, :])
    ax[1,0].grid()
    ax[1,0].set_title(f'width')
    ax[1,1].set_title(f'color_map')

    ax[1,2].plot(new_angles[y, :])
    ax[1,2].grid()

    ax[1,2].set_title(f'new_angles')
    plt.show()

    # y = 60
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].imshow(signal)
    ax[0].set_title(f'signal')

    ax[1].plot(signal[y, :])
    ax[1].set_title(f'signal')
    ax[1].grid()
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].imshow(res)
    ax[0].set_title(f'noisy')

    ax[1].plot(res[y, :])
    ax[1].set_title(f'noisy signal')
    ax[1].grid()
    plt.show()
