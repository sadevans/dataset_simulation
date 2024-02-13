import numpy as np
import cv2
import matplotlib.pyplot as  plt
import yaml
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from src.Img import *
from src.Fig import Figure


class Generator():
    def __init__(self, flag_image: str, image_size: tuple, color_hole: float, color_back: float, signal_algo: str, num_images: int):
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
        self.flag_image = flag_image # {'exist', 'scratch'}
        self.image_size = image_size
        self.color_hole = color_hole
        self.color_back = color_back
        self.signal_algo = signal_algo
        self.num_images = num_images

        self.cpu_count = multiprocessing.cpu_count()
        # self.cpu_count = 4
        self.image_objects = []
        self.images = [Img(np.zeros(self.image_size, dtype=np.float32))]*num_images

        with open('src/configs/figures_bank.yaml', 'r') as file:
            self.figures_bank = yaml.safe_load(file)

    
        if self.flag_image == 'scratch':
            self.generate_images()


    def generate_images(self):
        start_multi_time_v1 = time.time()
        # with ThreadPoolExecutor(self.cpu_count) as executor:
        #     for image in executor.map(self.generate_single_image, self.images):
        #         plt.imshow(image.image)
        #         plt.show()

        #         self.image_objects.append(image)

        try:
            pool = multiprocessing.Pool(processes = self.cpu_count)
            for image in pool.map(self.generate_single_image, self.images):
                # plt.imshow(image.image)
                # plt.show()

                self.image_objects.append(image)

        finally:
            pool.close()
            pool.join()
        print('Processing time: {0} [sec]'.format(time.time() - start_multi_time_v1))


    def generate_single_image(self, image):
        iters = len(self.figures_bank['figures'])
        position = (0,0)

        while iters > 0:
            random_figure = random.choice(self.figures_bank['figures'])
            shape = random_figure['shape']
            height = random_figure['height']
            width = random_figure['width']
            border_width = random.randint(2, 21)
            position = self.choose_new_position(image.image, height, width, border_width)
            if position is None:
                iters -= 1
            if 'rectangle' in shape and position is not None: 
                corner_radius = random_figure['corner_radius']
                image.image = self.draw_rect(image, position, height, width, corner_radius, border_width)
            elif 'circle' in shape and position is not None:
                image.image = self.draw_circle(image, position, height, border_width)

        return image


    def choose_new_position(self, image, height, width, border_width):
        offset = border_width//2 + 10
        blacks = np.argwhere(image == 0)
        iters = 30
        position = None
        while iters > 0:
            position = tuple(blacks[random.randint(0, len(blacks))])
            if 255 not in image[position[1] - height//2 - offset:position[1]+height//2+offset, position[0] -width//2-offset:position[0]+width//2+offset] and \
            128 not in image[position[1] - height//2 - offset:position[1]+height//2+offset, position[0] -width//2-offset:position[0]+width//2+offset] and \
                position[0] - width//2 -offset > 0 and position[0]+width//2 +offset< image.shape[1] and position[1] - height//2 - offset>0 and position[1]+height//2+offset< image.shape[0]:
                break
            else:
                iters -= 1
        if iters == 0: position = None
        return position
    

    def draw_rect(self, image, position, height, width, corner_radius, border_width):
        temp = np.zeros_like(image.image, dtype=np.float32)
        temp = cv2.rectangle(temp, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
                (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, 20) 
        temp = cv2.rectangle(temp, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
                    (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, -1)
    
        cont = image.init_contour(temp)
        cv2.drawContours(image.image, cont, 0, 128, border_width*2)
        cv2.drawContours(temp, cont, 0, 128, border_width*2)
        bord_cont = image.init_contour(temp)
        image.image = cv2.rectangle(image.image, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
                (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, 20) 
        image.image = cv2.rectangle(image.image, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
                    (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, -1)
        

        self.compute_figure(image, Figure(cont[0], bord_cont[0], image.image))

        # image.figures.append(Figure(cont[0], bord_cont[0]))
        return image.image
    

    def draw_circle(self, image, position, height, border_width):
        temp = np.zeros_like(image.image, dtype=np.float32)

        cv2.circle(temp, position, height//2, 255, 2)
        cv2.circle(temp, position, height//2, 255, -1)
        cont = image.init_contour(temp)
        cv2.drawContours(temp, cont, 0, 128, border_width*2)
        bord_cont = image.init_contour(temp)
        cv2.drawContours(image.image, cont, 0, 128, border_width*2)
        cv2.circle(image.image, position, height//2, 255, 2)
        cv2.circle(image.image, position, height//2, 255, -1)

        self.compute_figure(image, Figure(cont[0], bord_cont[0], image.image))
        # image.figures.append(Figure(cont[0], bord_cont[0]))
        return image.image
    

    def compute_figure(self, image, fig):
        image.figures.append(fig)

        


