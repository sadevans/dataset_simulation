import numpy as np
import cv2
import matplotlib.pyplot as  plt
from skimage.draw import line
import random



class Figure():
    def __init__(self, generator, hole_contour, border_contour, image):
        self.generator = generator
        self.hole_contour = hole_contour.copy()
        self.border_contour = border_contour.copy()

        self.color_map_local = np.zeros_like(image, dtype=np.float32)
        self.angles_map_local = np.zeros_like(image, dtype=np.float32)
        self.width_map_local = np.zeros_like(image, dtype=np.float32)

        self.mask_figure_local = np.zeros_like(image, dtype=np.float32)

        self.dp2 = (1, 0)
        self.dp3 = (0, 1)

        self.make_locate_mask()

        # plt.imshow(self.mask_figure_local)
        # plt.show()


    def make_locate_mask(self):
        cv2.drawContours(self.mask_figure_local, [self.border_contour], 0, 128, -1)
        cv2.drawContours(self.mask_figure_local, [self.hole_contour], 0, 255, -1)

    def init_contour(self, image):
        cont, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont


    def compute_local_maps(self):
       self.width_map, self.angles_map, self.color_map_local = self.transform_bezier([self.border_contour], [self.hole_contour])


    def detect_cont(self, img):
        cont, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont

    def draw_gradient_line(self, img, start_point, points, colors, thickness=4):
        start = start_point
        for i in range(1, len(points) - 1):
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


    def transform_bezier(self, ext, int):
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
                        new_line = np.zeros(dist_*self.generator.pixel_size, dtype=np.float32)
                        _, y = self.bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, self.generator.resist_thickness, 100)

                        _, colors = self.bezier(new_line, np.linspace(0, 1, len(new_line)), 0.0, self.generator.color_back, self.generator.color_hole)

                        reshaped_y  = np.array(y).reshape(-1, self.generator.pixel_size)  # Разбиваем на подмассивы по self.generator.pixel_size элементов
                        averages_y = np.max(reshaped_y, axis=1)

                        reshaped_colors  = np.array(colors).reshape(-1, self.generator.pixel_size)  # Разбиваем на подмассивы по self.generator.pixel_size элементов
                        angles = np.arctan(np.abs(np.gradient(y)))
                        new_angl = angles[::self.generator.pixel_size]
                        reshaped_angls  = np.array(angles).reshape(-1, self.generator.pixel_size)  # Разбиваем на подмассивы по self.generator.pixel_size элементов
                        averages_angls = np.max(reshaped_angls, axis=1)
                        max_indices = np.argmax(reshaped_angls, axis=1)
                        averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]
                    
                        self.draw_gradient_line(self.color_map_local, point, discrete_line, averages_colors, thickness=2)
                        if dist_>40: self.draw_gradient_line(self.angles_map_local, point, discrete_line, averages_angls, thickness=3)
                        else:self.draw_gradient_line(self.angles_map_local, point, discrete_line, averages_angls, thickness=2)
                            

                    elif dist_ == 1:
                        y = [self.generator.resist_thickness-10]
                        averages_colors = [(self.generator.color_back - 2)]
                        angles = [np.arctan(y[0]/(dist_*self.generator.pixel_size))]
                        cv2.line(self.angles_map_local, point, nearest_point, np.deg2rad(89), 2)
                        cv2.line(self.color_map_local, point, nearest_point, (np.tan(np.deg2rad(89))*self.generator.pixel_size * 110)/self.generator.resist_thickness, 2)

                    cv2.line(self.width_map_local, point, nearest_point, dist_, 3)


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
                        new_line = np.zeros(dist_*self.generator.pixel_size, dtype=np.float32)
                        _, y = self.bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, self.generator.resist_thickness, 100)

                        _, colors = self.bezier(new_line, np.linspace(0, 1, len(new_line)), 255.0, self.generator.color_back, self.generator.color_hole)
                        
                        reshaped_y  = np.array(y).reshape(-1, self.generator.pixel_size)  # Разбиваем на подмассивы по self.generator.pixel_size элементов
                        averages_y = np.max(reshaped_y, axis=1)
                        reshaped_colors  = np.array(colors).reshape(-1, self.generator.pixel_size)  # Разбиваем на подмассивы по self.generator.pixel_size элементов
                        angles = np.arctan(np.abs(np.gradient(y)))
                        
                        reshaped_angls  = np.array(angles).reshape(-1, self.generator.pixel_size)  # Разбиваем на подмассивы по self.generator.pixel_size элементов
                        averages_angls = np.max(reshaped_angls, axis=1)
                        max_indices = np.argmax(reshaped_angls, axis=1)
                        averages_colors = reshaped_colors[np.arange(len(reshaped_colors)), max_indices]

                    elif dist_ == 1:
                        y = [self.generator.resist_thickness-10]
                        averages_colors = [(self.generator.color_back - 2)]
                        angles = [np.arctan(y[0]/(dist_*self.generator.pixel_size))]
                        cv2.line(self.angles_map_local, point, nearest_point, np.deg2rad(89), 2)
                        cv2.line(self.color_map_local, point, nearest_point, (np.tan(np.deg2rad(89))*self.generator.pixel_size * 110)/self.generator.resist_thickness, 2)
                    
                    if self.angles_map_local[point[1], point[0]]  == 0:
                        self.draw_gradient_line(self.angles_map_local, point, discrete_line[-1::-1], averages_angls[-1::-1], thickness=2)

                    if self.color_map_local[point[1], point[0]]  == 0:
                        self.draw_gradient_line(self.color_map_local, point, discrete_line[-1::-1], averages_colors[-1::-1], thickness=2)

                    cv2.line(self.width_map_local, point, nearest_point, dist_, 2)


        img_cp = self.mask_figure_local.copy()
        mask = img_cp != 128 
        self.width_map_local[mask] = 0
        self.angles_map_local[mask] = 0
        self.color_map_local[self.mask_figure_local == 0] = self.generator.color_back
        self.color_map_local[self.mask_figure_local == 255] = self.generator.color_hole

        zeros = np.where((self.color_map_local==0) & (self.mask_figure_local==128))
        if len(zeros[0]) > 0:
            tmp = np.zeros_like(self.mask_figure_local)
            cv2.drawContours(tmp, [ext[0]], -1, 255, 0)
            c = self.detect_cont(tmp)
            ext = np.argwhere(tmp > 0)
            ext= np.array([list(reversed(ex)) for ex in ext])
            
            for i in range(len(zeros[0])):
                point = (zeros[0][i], zeros[1][i])

                index_int, dist_int = self.closest_point(point, cont_int)
                index_ext, dist_ext = self.closest_point(point, ext)
                nearest_point_int = cont_int[index_int]
                nearest_point_ext = ext[index_ext]
                discrete_line_int = list(zip(*line(*point, *nearest_point_int)))
                discrete_line_ext = list(zip(*line(*point, *nearest_point_ext)))
                distance_int = len(discrete_line_int)
                distance_ext = len(discrete_line_ext)
                if distance_int < distance_ext:
                    val = self.generator.color_back - distance_int*np.tan(self.angles_map_local[point[0], point[1]])/(distance_ext + distance_int)
                else:
                    val = self.generator.color_hole + distance_ext*np.tan(self.angles_map_local[point[0], point[1]])/(distance_ext + distance_int)

                if val == 85.0:
                    val = (110.0 + 85.0)/2 -random.randint(10, 14)
                elif val == 110.0:
                    val = (110.0 + 85.0)/2 -random.randint(10, 14)
                self.color_map_local[point[0], point[1]] = np.abs(val)
        # self.color_map_local[self.mask_figure_local == 0] = 0
        # self.color_map_local[self.mask_figure_local == 255] = 0
        return self.width_map_local, self.angles_map_local, self.color_map_local

