import numpy as np
import cv2
import matplotlib.pyplot as  plt
import random
import yaml
import torch
from torch.distributions.gamma import Gamma


from src.Fig import *


class Img():
    def __init__(self, solver, image:np.array):
        self.solver = solver
        self.image = image.copy()
        self.figures = []

        self.color_map = np.zeros_like(image, dtype=np.float32)
        self.angles_map = np.zeros_like(image, dtype=np.float32)
        self.width_map = np.zeros_like(image, dtype=np.float32)

        self.signal_image = np.zeros_like(image, dtype=np.float32)
        self.raw_image = self.image.copy()
        self.E = 10

        # по хорошему передавать в класс
        with open('src/configs/figures_bank.yaml', 'r') as file:
            self.figures_bank = yaml.safe_load(file)


    def init_contour(self, image):
        cont, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont
    


    def place_single_figure(self, iters, position):
        random_figure = random.choice(self.figures_bank['figures'])
        shape = random_figure['shape']
        height = random_figure['height']
        width = random_figure['width']
        border_width = random.randint(2, 21)
        position = self.choose_new_position(self.image, height, width, border_width)
        
        if position is None:
            iters -= 1
        else:
            fig = Figure(position, None, None, self.image)
            self.image = fig.draw(self.image, random_figure, position, border_width)
            self.figures.append(fig)


        return

    def paste_locals_in_global(self, pixels, figure):
        self.color_map[pixels] = figure.color_map_local[pixels].astype(np.float32)
        self.angles_map[pixels] = figure.angles_map_local[pixels].astype(np.float32)
        self.width_map[pixels] = figure.width_map_local[pixels].astype(np.float32)

    def place_figures(self):
        """
        place figures in an image here
        """

        iters = len(self.figures_bank['figures'])
        position = (0,0)
        while iters > 0:
            random_figure = random.choice(self.figures_bank['figures'])
            shape = random_figure['shape']
            height = random_figure['height']
            width = random_figure['width']
            border_width = random.randint(2, 21)
            position = self.choose_new_position(height, width, border_width)
            
            if position is None:
                iters -= 1
            else:
                fig = Figure(position, [], [], self.image)
                self.image = fig.draw(self.image, random_figure, position, border_width)
                self.figures.append(fig)
                fig.compute_local_maps(self.solver)
                figure_pixels = np.where(fig.mask_figure_local != 0)

                self.paste_locals_in_global(figure_pixels, fig)

        self.color_map[self.image==255] = 85.0
        self.color_map[self.image==0] = 110.0

        self.compute_signal()
        self.add_noise()


    def choose_new_position(self, height, width, border_width):
    
        """
        возможно сделать так, чтобы из рандомной точки опредяляся максимальный bounding box 
        и вставлялась фигура под него

        или разбить изображение случайно на прямоугольники и под них подбирать
        """
        # print(type(self.image))
        offset = border_width//2 + 10
        blacks = np.argwhere(self.image == 0)
        iters = 30
        position = None
        while iters > 0:
            position = tuple(blacks[random.randint(0, len(blacks))])
            if 255 not in self.image[position[1] - height//2 - offset:position[1]+height//2+offset, position[0] -width//2-offset:position[0]+width//2+offset] and \
            128 not in self.image[position[1] - height//2 - offset:position[1]+height//2+offset, position[0] -width//2-offset:position[0]+width//2+offset] and \
                position[0] - width//2 -offset > 0 and position[0]+width//2 +offset< self.image.shape[1] \
                and position[1] - height//2 - offset>0 and position[1]+height//2+offset< self.image.shape[0]:
                break
            else:
                iters -= 1
        if iters == 0: position = None
        return position 
        


    def compute_signal(self):
        alpha_bord = self.angles_map[self.image == 128].copy()
        alpha_bord[alpha_bord==0.0] = np.radians(1)

        alpha_back = self.angles_map[self.image == 0].copy()
        alpha_hole = self.angles_map[self.image == 255].copy()

        self.signal_image[self.image == 0] =  (self.E / (np.abs(np.cos(np.radians(alpha_back + 1)))**(0.87))) + self.color_map[self.image==0]
        self.signal_image[self.image == 128] = (self.E / (np.abs(np.cos(np.radians(90) - (np.radians(180 - 90) - alpha_bord)))**(0.87))) + \
            self.color_map[self.image==128]
        self.signal_image[self.image == 255] = (self.E / (np.abs(np.cos(np.radians(alpha_hole + 1)))**(1.1))) + self.color_map[self.image==255]

        self.signal_image = np.clip(self.signal_image, 0, 255)

        self.signal_image = np.clip(cv2.GaussianBlur(self.signal_image, (11,11), 0), 0, 255)

        cv2.imwrite('/home/sadevans/space/work/dataset_simulation/test_version/data/test_signal.png', self.signal_image.astype(np.uint8))


    def add_noise(self):
        a = 30
        b = 6
        m = Gamma(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))

        self.raw_image = (torch.Tensor(self.signal_image) + m.sample()* torch.randn(torch.Tensor(self.signal_image).shape)).numpy().copy()
        cv2.imwrite('/home/sadevans/space/work/dataset_simulation/test_version/data/test_raw.png', self.raw_image.astype(np.uint8))

