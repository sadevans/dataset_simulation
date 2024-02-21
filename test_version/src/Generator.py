import numpy as np
import cv2
import matplotlib.pyplot as  plt
import yaml
import random
import multiprocessing
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import threading
import queue
import time
from src.Img import *
from src.Fig import Figure
import itertools
import gc
import streamlit as st
# from streamlit import caching

# POOL_SIZE = 4
# local_storage = threading.local()


class Generator():
    def __init__(self, solver, flag_image: str, image_size: tuple, color_hole: float, color_back: float, signal_algo: str, num_images: int, num_figures: int, path: os.path):
        """
        - [ ]  изображения с нуля или же по существующим бинарным маскам
        - [ ]  для генерации с нуля размер изображений задается пользователем
        - [ ]  цвета фона и дна задаются пользователем
        - [ ]  алгоритм расчета границы задается пользователем
        - [ ]  алгоритм расчета сигнала задается пользователем
        - [ ]  размер пискеля азадется пользователем
        - [ ]  толщина резиста задается пользователем
        - [ ]  название папки с изображениями задается пользователем
        - [ ]  расположение папки с изображениями задается пользователем
        """
        self.solv = solver
        self.flag_image = flag_image # {'exist', 'scratch'}
        self.image_size = image_size
        self.color_hole = color_hole
        self.color_back = color_back
        self.signal_algo = signal_algo
        self.num_images = num_images
        self.num_figures = num_figures
        self.root_path = path

        self.pixel_size = 12
        self.resist_thickness = 700

        # self.cpu_count = multiprocessing.cpu_count()
        self.cpu_count = 4
        self.image_objects = []
        # self.images = [Img(np.zeros(self.image_size, dtype=np.float32))]*num_images

        with open('src/configs/figures_bank.yaml', 'r') as file:
            self.figures_bank = yaml.safe_load(file)

    
        if self.flag_image == 'scratch':
            self.generate_images()

    def generate_single_image_star(self, a_b):
        return self.generate_single_image(*a_b)
    

    def generate_single_image(self, i):
        img = Img(self.solv, np.zeros(self.image_size, dtype=np.float32))
        img.place_figures(num_figures = 100)
        return img


    def generate_images(self):
        # start_multi_time_v1 = time.time()

        # resources = queue.Queue(POOL_SIZE)  # one resource per thread
        # for pool_idx in range(POOL_SIZE):
        #     resources.put(pool_idx, False)
        # seeds = list(np.arange(1, len(self.images)))
        # i = 0
        # pool =  ThreadPool(8)
        # for img in pool.map(self.generate_single_image, np.arange(0,self.num_images)):
        #     self.image_objects.append(img)
        #     cv2.imwrite(f'/home/sadevans/space/work/dataset_simulation/test_version/data/masks/mask_{i}.png', img.image.astype(np.uint8))
        #     i += 1
        # pool =  ThreadPool(4)
        # # for i, image in enumerate(self.images):
        # for i in range(self.num_images):

        #     # print(type(self.images[i]))
        #     # img = pool.apply_async(self.generate_single_image, [image]).get()
        #     img = pool.apply_async(self.generate_single_image, [i]).get()

        #     # print(type(img))
        #     self.image_objects.append(img)
        #     cv2.imwrite(f'/home/sadevans/space/work/dataset_simulation/test_version/data/img_{i}_{i}.png', img.image.astype(np.uint8))
        # pool.close()
        # pool.join()

        try:
            pool = multiprocessing.Pool(processes = 4)
            i = 0
            # for image in pool.map(self.generate_single_image, self.images):
            for image in pool.map(self.generate_single_image, np.arange(0,self.num_images)):
                i = len(os.listdir(f'{self.root_path}/semantic_masks/'))
                plt.imshow(image.image)
                plt.show()
                gc.collect()

                cv2.imwrite(f'{self.root_path}/semantic_masks/{i:04d}.png', image.image.astype(np.uint8))
                cv2.imwrite(f'{self.root_path}/signal/{i:04d}.png', image.signal_image.astype(np.uint8))
                cv2.imwrite(f'{self.root_path}/raw/{i:04d}.png', image.raw_image.astype(np.uint8))


                # del image, i
                del image, i
                gc.collect()
                # caching.clear_cache()
                # st.runtime.legacy_caching.clear_cache()
                # gc.collect()
                # i+=1
        finally:
            pool.close()
            pool.join()
            del pool
            gc.collect()
        # print('Processing time: {0} [sec]'.format(time.time() - start_multi_time_v1))


    # def generate_single_image(self, image, seed_num):
    # def generate_single_image(self, image):
    #     # random.seed(seed_num)
    #     iters = len(self.figures_bank['figures'])
    #     position = (0,0)
    #     k = 0
    #     while iters > 0:
    #         random_figure = random.choice(self.figures_bank['figures'])
    #         shape = random_figure['shape']
    #         height = random_figure['height']
    #         width = random_figure['width']
    #         border_width = random.randint(2, 21)
    #         position = self.choose_new_position(image.image, height, width, border_width)
    #         # if k ==0:
    #             # print('position: ', position)
            
    #         if position is None:
    #             iters -= 1
    #         if 'rectangle' in shape and position is not None: 
    #             corner_radius = random_figure['corner_radius']
    #             image.image = self.draw_rect(image, position, height, width, corner_radius, border_width)
    #         elif 'circle' in shape and position is not None:
    #             image.image = self.draw_circle(image, position, height, border_width)
    #         k+= 1


    #     return image


    # def choose_new_position(self, image, height, width, border_width):
    #     offset = border_width//2 + 10
    #     blacks = np.argwhere(image == 0)
    #     iters = 30
    #     position = None
    #     while iters > 0:
    #         position = tuple(blacks[random.randint(0, len(blacks))])
    #         if 255 not in image[position[1] - height//2 - offset:position[1]+height//2+offset, position[0] -width//2-offset:position[0]+width//2+offset] and \
    #         128 not in image[position[1] - height//2 - offset:position[1]+height//2+offset, position[0] -width//2-offset:position[0]+width//2+offset] and \
    #             position[0] - width//2 -offset > 0 and position[0]+width//2 +offset< image.shape[1] and position[1] - height//2 - offset>0 and position[1]+height//2+offset< image.shape[0]:
    #             break
    #         else:
    #             iters -= 1
    #     if iters == 0: position = None
    #     return position
    

    # def draw_rect(self, image, position, height, width, corner_radius, border_width):
    #     temp = np.zeros_like(image.image, dtype=np.float32)
    #     temp = cv2.rectangle(temp, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
    #             (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, 20) 
    #     temp = cv2.rectangle(temp, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
    #                 (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, -1)
    
    #     cont = image.init_contour(temp)
    #     cv2.drawContours(image.image, cont, 0, 128, border_width*2)
    #     cv2.drawContours(temp, cont, 0, 128, border_width*2)
    #     bord_cont = image.init_contour(temp)
    #     image.image = cv2.rectangle(image.image, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
    #             (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, 20) 
    #     image.image = cv2.rectangle(image.image, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
    #                 (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, -1)
        

    #     self.compute_figure(image, Figure(self, cont[0].reshape(-1, 2), bord_cont[0].reshape(-1, 2), image.image))

    #     # image.figures.append(Figure(cont[0], bord_cont[0]))
    #     return image.image
    

    # def draw_circle(self, image, position, height, border_width):
    #     temp = np.zeros_like(image.image, dtype=np.float32)

    #     cv2.circle(temp, position, height//2, 255, 2)
    #     cv2.circle(temp, position, height//2, 255, -1)
    #     cont = image.init_contour(temp)
    #     cv2.drawContours(temp, cont, 0, 128, border_width*2)
    #     bord_cont = image.init_contour(temp)
    #     cv2.drawContours(image.image, cont, 0, 128, border_width*2)
    #     cv2.circle(image.image, position, height//2, 255, 2)
    #     cv2.circle(image.image, position, height//2, 255, -1)

    #     # print(cont[0])
    #     self.compute_figure(image, Figure(self, cont[0].reshape(-1, 2), bord_cont[0].reshape(-1, 2), image.image))
    #     # image.figures.append(Figure(cont[0], bord_cont[0]))
    #     return image.image
    

    def compute_figure(self, image, fig):
        image.figures.append(fig)
        fig.compute_local_maps()

        # print(fig.hole_contour)
        

        


