import numpy as np
import cv2
import matplotlib.pyplot as  plt
import torch
from torch.distributions.gamma import Gamma


class Img():
    def __init__(self, image:np.array):
        self.image = image.copy()
        self.figures = []

        self.color_map = np.zeros_like(image, dtype=np.float32)
        self.angles_map = np.zeros_like(image, dtype=np.float32)
        self.width_map = np.zeros_like(image, dtype=np.float32)

        self.signal_image = np.zeros_like(image, dtype=np.float32)
        self.raw_image = self.image.copy()


    def init_contour(self, image):
        cont, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont


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


    def add_noise(self):
        a = 30
        b = 6
        m = Gamma(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))

        self.raw_image = (torch.Tensor(self.signal_image) + m.sample()* torch.randn(torch.Tensor(self.signal_image).shape)).copy()

