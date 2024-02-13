import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import multiprocessing
import time
from skimage.draw import line


class Solver():
    def __init__(self, signal_formula: str, pixel_size: int, resist_thickness: int, k: float, E: float,\
                 dp2:tuple, dp3:tuple):
        # self.parent_ = parent
        self.signal_formula = signal_formula
        self.pixel_size = pixel_size
        self.resist_thickness = resist_thickness


        self.color_back = 110.0
        self.color_hole = 85.0
        self.k = k
        self.E = E

        self.dp2 = dp2
        self.dp3 = dp3

        self.i = 0

        self.cpu_count = multiprocessing.cpu_count()
        # self.process()


    def draw_gradient_line(self, img, start_point, points, colors, thickness=4):
        start = start_point
        for i in range(1, len(points) - 1):
            # if img[start[1], start[0]] == 0 or img[start[1], start[0]] == 255 or img[start[1], start[0]]==128:
            if i+1 != len(points)-1:
                cv2.line(img, start, points[i+1], colors[i], thickness)
            start = points[i]
    


    def closest_point(self, point, array):
        diff = array - point
        distance = np.einsum('ij,ij->i', diff, diff)
        return np.argmin(distance), distance
    

    def compute_previous_pixel(self, first_pixel, last_pixel, distance=1):
        x1, y1 = first_pixel
        x2, y2 = last_pixel

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        length = 1 if length == 0 else length
        t = -distance / length
        x_t = np.round((x1 + t * (x2 - x1)), 0).astype(np.int32)
        y_t = np.round((y1 + t * (y2 - y1)), 0).astype(np.int32)

        return (x_t, y_t)
    

    def compute_next_pixel(self, first_point, last_point, distance=1):
        x1, y1 = first_point
        x2, y2 = last_point

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        t = 1 + distance / length
        x_t = np.round((x1 + t * (x2 - x1)), 0).astype(np.int32)
        y_t = np.round((y1 + t * (y2 - y1)), 0).astype(np.int32)

        return (x_t, y_t)
    

    def detect_cont(self, img):
        cont, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont
    
    # @lru_cache
    def bezier(self, line, t, prev, color_back, color_hole):
        if prev == 0.0:
            point1 = (0, color_back)
            point4 = (len(line), color_hole)
            point2 = (self.dp3[0]*len(line), self.dp3[1]*(color_back - color_hole) + color_hole) # default: (len(line), color_back)
            point3 = (self.dp2[0]*len(line), self.dp2[1]*(color_back - color_hole) + color_hole) #default: (0, color_hole)

        if prev == 255.0:
            point1 = (0, color_hole)
            point4 = (len(line), color_back)
            point3 = (self.dp3[0]*len(line), self.dp3[1]*(color_back - color_hole) + color_hole) # dafault: (0, color_back)
            point2 = (self.dp2[0]*len(line), self.dp2[1]*(color_back - color_hole) + color_hole) # default: (len(line), color_hole)
        x = point1[0]*(1-t)**3 + point2[0]*3*t*(1-t)**2 + point3[0]*3*t**2*(1-t) + point4[0]*t**3
        vals = point1[1]*(1-t)**3 + point2[1]*3*t*(1-t)**2 + point3[1]*3*t**2*(1-t) + point4[1]*t**3
        return x, vals


    def transform_bezier(self, img, ext, int):
        width_img = np.zeros_like(img, dtype=np.float32)
        new_angles = np.zeros_like(img, dtype=np.float32)
        color_map = np.zeros_like(img, dtype=np.float32)

        for cont_ext, cont_int in zip(ext, int):
            for point in cont_ext:
                    min_dist = float('inf')
                    index, dist = self.closest_point(point, cont_int)
                    if dist[index] < min_dist :
                        min_dist = dist[index].item()
                        nearest_point = cont_int[index] 
                        prev = [self.compute_previous_pixel(point, nearest_point)]
                        discrete_line = list(zip(*line(*prev[0], *nearest_point))) # find all pixels from the line
                        dist_ = len(discrete_line) - 2

                    if dist_ > 1:
                        new_line = np.zeros(dist_*self.pixel_size, dtype=np.float32)
                        _, y = self.bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, self.resist_thickness, 100)
                        _, colors = self.bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, self.color_back, self.color_hole)

                        reshaped_y  = np.array(y).reshape(-1, self.pixel_size)  # Разбиваем на подмассивы по self.pixel_size элементов
                        averages_y = np.max(reshaped_y, axis=1)

                        reshaped_colors  = np.array(colors).reshape(-1, self.pixel_size)  # Разбиваем на подмассивы по self.pixel_size элементов
                        angles = np.arctan(np.abs(np.gradient(y)))
                        # if dist_==2:print(angles)
                        new_angl = angles[::self.pixel_size]
                        reshaped_angls  = np.array(angles).reshape(-1, self.pixel_size)  # Разбиваем на подмассивы по self.pixel_size элементов
                        averages_angls = np.max(reshaped_angls, axis=1)
                        max_indices = np.argmax(reshaped_angls, axis=1)
                        averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]
                    
                        self.draw_gradient_line(color_map, point, discrete_line, averages_colors, thickness=2)
                        if dist_>40: self.draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=3)
                        else:self.draw_gradient_line(new_angles, point, discrete_line, averages_angls, thickness=2)
                            

                    elif dist_ == 1:
                        y = [self.resist_thickness-10]
                        averages_colors = [(self.color_back - 2)]
                        angles = [np.arctan(y[0]/(dist_*self.pixel_size))]
                        cv2.line(new_angles, point, nearest_point, np.deg2rad(89), 2)
                        cv2.line(color_map, point, nearest_point, (np.tan(np.deg2rad(89))*self.pixel_size * 110)/self.resist_thickness, 2)

                    cv2.line(width_img, point, nearest_point, dist_, 3)


        for cont_ext, cont_int in zip(ext, int):
            for point in cont_int:
                    min_dist = float('inf')
                    index, dist = self.closest_point(point, cont_ext)
                    if dist[index] < min_dist :
                        min_dist = dist[index].item()
                        nearest_point = cont_ext[index]
                        next = [self.compute_next_pixel(point, nearest_point)]
                        discrete_line = list(zip(*line(*point, *next[0]))) # find all pixels from the line
                        dist_ = len(discrete_line) - 2

                    if dist_ == 0:
                        dist_ = 1

                    if dist_ > 1:
                        new_line = np.zeros(dist_*self.pixel_size, dtype=np.float32)
                        _, y = self.bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, self.resist_thickness, 100)
                        _, colors = self.bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, self.color_back, self.color_hole)
                        reshaped_y  = np.array(y).reshape(-1, self.pixel_size)  # Разбиваем на подмассивы по self.pixel_size элементов
                        averages_y = np.max(reshaped_y, axis=1)
                        reshaped_colors  = np.array(colors).reshape(-1, self.pixel_size)  # Разбиваем на подмассивы по self.pixel_size элементов
                        angles = np.arctan(np.abs(np.gradient(y)))
                        reshaped_angls  = np.array(angles).reshape(-1, self.pixel_size)  # Разбиваем на подмассивы по self.pixel_size элементов
                        averages_angls = np.max(reshaped_angls, axis=1)
                        max_indices = np.argmax(reshaped_angls, axis=1)
                        averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]

                    elif dist_ == 1:
                        y = [self.resist_thickness-10]
                        averages_colors = [(self.color_back - 2)]
                        angles = [np.arctan(y[0]/(dist_*self.pixel_size))]
                        cv2.line(new_angles, point, nearest_point, np.deg2rad(89), 2)
                        cv2.line(color_map, point, nearest_point, (np.tan(np.deg2rad(89))*self.pixel_size * 110)/self.resist_thickness, 2)
                    
                    if new_angles[point[1], point[0]]  == 0:
                        self.draw_gradient_line(new_angles, point, discrete_line[-1::-1], averages_angls[-1::-1], thickness=2)
                        
                    if color_map[point[1], point[0]]  == 0:
                        self.draw_gradient_line(color_map, point, discrete_line[-1::-1], averages_colors[-1::-1], thickness=2)
                    cv2.line(width_img, point, nearest_point, dist_, 2)


        img_cp = img.copy()
        mask = img_cp != 128 
        width_img[mask] = 0
        new_angles[mask] = 0
        color_map[img == 0] = self.color_back
        color_map[img == 255] = self.color_hole

        zeros = np.where((color_map==0) & (img==128))
        if len(zeros[0]) > 0:
            tmp = np.zeros_like(img)
            cv2.drawContours(tmp, [ext[0]], -1, 255, 0)
            c = self.detect_cont(tmp)
            ext = np.argwhere(tmp > 0)
            ext = np.array([list(reversed(ex)) for ex in ext])
            
            for i in range(len(zeros[0])):
                point = (zeros[0][i], zeros[1][i])

                index_int, _ = self.closest_point(point, cont_int)
                index_ext, _ = self.closest_point(point, ext)
                nearest_point_int = cont_int[index_int]
                nearest_point_ext = ext[index_ext]
                discrete_line_int = list(zip(*line(*point, *nearest_point_int)))
                discrete_line_ext = list(zip(*line(*point, *nearest_point_ext)))
                distance_int = len(discrete_line_int)
                distance_ext = len(discrete_line_ext)
                if distance_int < distance_ext:
                    val = self.color_back - distance_int*np.tan(new_angles[point[0], point[1]])/(distance_ext + distance_int)
                else:
                    val = self.color_hole + distance_ext*np.tan(new_angles[point[0], point[1]])/(distance_ext + distance_int)

                if val == 85.0:
                    val = (110.0 + 85.0)/2 -random.randint(10, 14)
                elif val == 110.0:
                    val = (110.0 + 85.0)/2 -random.randint(10, 14)
                color_map[point[0], point[1]] = np.abs(val)
        # color_map[img == 0] = 0
        # color_map[img == 255] = 0
        color_map[img == 0] = self.color_back
        color_map[img == 255] = self.color_hole
        return width_img, new_angles, color_map
    

    