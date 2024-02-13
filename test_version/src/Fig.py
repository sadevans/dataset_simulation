import numpy as np
import cv2
import matplotlib.pyplot as  plt


class Figure():
    def __init__(self, hole_contour, border_contour, image):
        self.hole_contour = hole_contour.copy()
        self.border_contour = border_contour.copy()

        self.mask_figure = np.zeros_like(image, dtype=np.float32)

        cv2.drawContours(self.mask_figure, self.border_contour, 0, 128, -1)
        cv2.drawContours(self.mask_figure, self.hole_contour, 0, 255, -1)
        print(np.unique(self.mask_figure))


    def init_contour(self, image):
        cont, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont

