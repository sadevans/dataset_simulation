import numpy as np
import cv2
import matplotlib.pyplot as  plt
import random
import yaml
import torch
from torch.distributions.gamma import Gamma
import gc
import torch
import os
from pathlib import Path
import glob
from src.Fig import *


class Img():
    def __init__(self, solver, image:np.array):
        self.solver = solver
        self.image = image.copy()
        self.figures = []

        self.color_map = np.zeros_like(image, dtype=np.float32)
        self.color_map[self.color_map==0] = self.solver.color_back
        self.angles_map = np.zeros_like(image, dtype=np.float32)
        self.width_map = np.zeros_like(image, dtype=np.float32)

        self.signal_image = np.zeros_like(image, dtype=np.float32)
        self.raw_image = self.image.copy()
        self.E = 10

        db_dir = '/home/sadevans/space/work/dataset_simulation/test_version/db/'
        # по хорошему передавать в класс
        # with open('src/configs/figures_bank.yaml', 'r') as file:
        #     self.figures_bank = yaml.safe_load(file)

        # self.figures_bank_ = [file.name for file in Path("/home/sadevans/space/work/dataset_simulation/test_version/db/").glob("*.png")]
        self.figures_bank = [db_dir + x for x in os.listdir(db_dir)]
        # print(self.figures_bank_)


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


    def place_figures(self, num_figures):
        figure_type = 'square' # default figure type

        for _ in range(num_figures):
            figure_type = random.choice(['circle', 'square', 'vertical line', 'horizontal line', 'vertical lines array', 'horizontal lines array'])
            border_width = random.randint(2, 20)
            figure_width = random.randint(10, 500)
            figure_height = random.randint(10, 500)
            if figure_type == 'square' or figure_type == 'circle':
                figure_height = max(figure_height, figure_width)
                figure_width = figure_height
            elif 'vertical line' in figure_type:
                if figure_width == figure_height:
                    figure_width = figure_height//3
            elif 'horizontal line' in figure_type:
                if figure_width == figure_height:
                    figure_height = figure_width//3

            x = random.randint(0, self.image.shape[1] - figure_width)
            y = random.randint(0, self.image.shape[0] - figure_height)
            new_figure = ((x, y), figure_width, figure_height)
            fig = Figure((x, y), [], [], self.image)
            count_overlap = 0
            while self.does_overlap(new_figure, border_width):
                if count_overlap > 10:
                    k = random.randint(0, 1)
                    if k == 0:
                        if figure_height >= 20: figure_height = figure_height - 10
                        if figure_width >= 20: figure_width = figure_width - 10
                    else:
                        if border_width >= 4: border_width -= 2

                x = random.randint(0, self.image.shape[1] - figure_width)
                y = random.randint(0, self.image.shape[0] - figure_height)

                new_figure = ((x, y), figure_width, figure_height)
                fig = Figure((x, y), [], [], self.image)
                count_overlap += 1
                if count_overlap == 50:
                    break
            
            if count_overlap < 50:
                if 'array' in figure_type:
                    continue
                elif figure_type == 'square' or 'line' in figure_type:
                    self.image = fig.draw_rect(self.image, new_figure, 20, border_width)
                elif figure_type == 'circle':
                    self.image = fig.draw_circle(self.image, new_figure, border_width)
                fig.compute_local_maps(self.solver)

                fig.make_flash(self.solver)
                figure_pixels = np.where(fig.mask_figure_local != 0)

                self.paste_locals_in_global(figure_pixels, fig)

                del fig
                gc.collect()
        
        self.compute_signal()
        self.add_noise()

        


    def does_overlap(self, new_figure, border_width):
        offset = 10 + border_width//2
        if (new_figure[0][0] >= self.image.shape[1]) or (new_figure[0][1] >= self.image.shape[0]) or (new_figure[0][0] - new_figure[1]//2 - offset <= 0) or (new_figure[0][1] - new_figure[2]//2 - offset <= 0):
            return True

        elif (255 not in self.image[new_figure[0][1] - new_figure[2]//2 - offset:new_figure[0][1] + (new_figure[2] - new_figure[2]//2) + offset, new_figure[0][0] - new_figure[1]//2 - offset:new_figure[0][0] + (new_figure[1] - new_figure[1]//2) + offset]) and \
            (128 not in self.image[new_figure[0][1] - new_figure[2]//2 - offset:new_figure[0][1] + (new_figure[2] - new_figure[2]//2) + offset, new_figure[0][0] - new_figure[1]//2 - offset:new_figure[0][0] + (new_figure[1] - new_figure[1]//2) + offset]) or \
            (new_figure[0][1] + (new_figure[2] - new_figure[2]//2) >= self.image.shape[0]) or (new_figure[0][0] + (new_figure[1] - new_figure[1]//2) >= self.image.shape[1]):
            return False
        else: 
            return True
        

    # def place_figures(self):
    #     """
    #     place figures in an image here
    #     """

    #     # iters = len(self.figures_bank['figures'])
    #     iters = len(self.figures_bank)

    #     position = (0,0)
    #     while iters > 0:
    #         # random_figure = random.choice(self.figures_bank['figures'])
    #         random_figure = random.choice(self.figures_bank)
    #         # print(random_figure)
    #         patch = cv2.imread(random_figure, 0).astype(np.float32)
    #         # shape = random_figure['shape']
    #         # height = random_figure['height']
    #         # width = random_figure['width']
    #         border_width = random.randint(2, 21)
    #         # position = self.choose_new_position(height, width, border_width)
    #         position = self.choose_new_position(patch, border_width)

            
    #         if position is None:
    #             iters -= 1
    #         else:
    #             fig = Figure(position, [], [], self.image)
    #             # self.image = fig.draw(self.image, random_figure, position, border_width)
    #             self.image = fig.paste(self.image, patch, position, border_width)

    #             # self.figures.append(fig)
    #             fig.compute_local_maps(self.solver)

    #             fig.make_flash(self.solver)
    #             figure_pixels = np.where(fig.mask_figure_local != 0)

    #             self.paste_locals_in_global(figure_pixels, fig)

    #             del fig
    #             gc.collect()


    #     self.compute_signal()
    #     self.add_noise()


    
    def choose_new_position(self, patch, border_width):
    
        """
        возможно сделать так, чтобы из рандомной точки опредяляся максимальный bounding box 
        и вставлялась фигура под него

        или разбить изображение случайно на прямоугольники и под них подбирать
        """
        # temp = np.zeros(patch.shape)
        # if temp in self.image:
        #     print('HERE')
        offset = border_width//2 + 6
        blacks = np.argwhere(self.image == 0)
        iters = 30
        position = None
        minimum = 0
        while iters > 0:
            # position = tuple(blacks[random.randint(0, len(blacks))])
            position = (random.randint(0, self.image.shape[0]), random.randint(minimum, self.image.shape[1]))
            if (position[0]-patch.shape[1]//2 -offset) > 0 and (position[0]+patch.shape[1]//2+offset) < self.image.shape[1] and (position[1]-patch.shape[0]//2-offset) > 0 \
            and (position[1]+patch.shape[0]//2+offset) < self.image.shape[0] and 255 not in \
                self.image[position[1]-patch.shape[0]//2 - offset:position[1]+patch.shape[0] - patch.shape[0]//2+offset, position[0]-patch.shape[1]//2-offset:position[0]+patch.shape[1] - patch.shape[1]//2+offset] and 128 not in \
                self.image[position[1]-patch.shape[0]//2 - offset:position[1]+patch.shape[0]-patch.shape[0]//2+offset, position[0]-patch.shape[1]//2-offset:position[0]+patch.shape[1] - patch.shape[1]//2+offset]:
                break
            # if 255 not in \
            #     self.image[position[1]-patch.shape[0]//2 - offset:position[1]+patch.shape[0] - patch.shape[0]//2+offset, position[0]-patch.shape[1]//2-offset:position[0]+patch.shape[1] - patch.shape[1]//2+offset] and 128 not in \
            #     self.image[position[1]-patch.shape[0]//2 - offset:position[1]+patch.shape[0]-patch.shape[0]//2+offset, position[0]-patch.shape[1]//2-offset:position[0]+patch.shape[1] - patch.shape[1]//2+offset]:
            #     break
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


    def add_noise(self):
        a = random.randint(50, 70)
        b = random.randint(3, 8)
        # b = 3
        m = Gamma(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))

        self.raw_image = np.clip((torch.Tensor(self.signal_image) + m.sample()* torch.randn(torch.Tensor(self.signal_image).shape)).numpy().copy(), 0, 255)

