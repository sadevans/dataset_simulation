import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import multiprocessing
import time
from skimage.draw import line
import gc


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
    

    def exponential_percent_rule(self, dist_, percent, growth_rate):
        new_line = np.array([percent * growth_rate ** i  for i in range(dist_)])
        new_line[new_line<0] = 0.0
        return new_line


    def linear_percent_rule(self, dist_, percent, step):
        new_line = np.linspace(percent, percent+step*dist_, dist_)
        new_line[new_line<0] = 0.0
        return new_line


    def bezier_percent_rule(self, dist_, percent, growth_rate):
        t = np.linspace(0, 1, dist_)
        point1 = (0, percent//growth_rate)
        point4 = (dist_, percent//growth_rate)
        point3 = (dist_, percent//growth_rate)
        point2 = (0, percent)
        new_line = point1[1]*(1-t)**3 + point2[1]*3*t*(1-t)**2 + point3[1]*3*t**2*(1-t) + point4[1]*t**3
        new_line[new_line<0] = 0.0

        return new_line


    def bezier_parabola_percent_rule(self, dist_, percent, growth_rate):
        point1 = (0,percent//2)
        point4 = (dist_, percent)

        point3 = (dist_//2, percent)
        point2 = (dist_//2, percent//2)
        t = np.linspace(0, 1, dist_//2)
        vals = point1[1]*(1-t)**3 + point2[1]*3*t*(1-t)**2 + point3[1]*3*t**2*(1-t) + point4[1]*t**3
        return list(vals) + list(vals)[::-1]


    def bezier_parabola(self, line, t, color_hole, percent):
        point1 = (0, color_hole)
        new_color_hole = color_hole * (1 + percent/100)
        point4 = (len(line), new_color_hole)

        point3 = (len(line)//2, new_color_hole)
        point2 = (len(line)//2, color_hole)
        x = point1[0]*(1-t)**3 + point2[0]*3*t*(1-t)**2 + point3[0]*3*t**2*(1-t) + point4[0]*t**3
        vals = point1[1]*(1-t)**3 + point2[1]*3*t*(1-t)**2 + point3[1]*3*t**2*(1-t) + point4[1]*t**3

        return x, vals



    def compute_new_colors(self, dist_, percent):
        new_line = np.zeros(dist_*self.pixel_size, dtype=np.float32)
        _, new_colors = self.bezier_parabola(new_line, np.linspace(0, 1, len(new_line)), self.color_hole, percent)
        _, new_colors_back = self.bezier_parabola(new_line, np.linspace(0, 1, len(new_line)), self.color_back, percent*0.5)

        reshaped_new_colors  = np.array(new_colors).reshape(-1, self.pixel_size)
        new_angles = np.arctan(np.abs(np.gradient(new_colors)))
        reshaped_new_angls  = np.array(new_angles).reshape(-1, self.pixel_size)

        averages_angls = np.max(reshaped_new_angls, axis=1)
        max_indices = np.argmax(reshaped_new_angls, axis=1)
        averages_colors = reshaped_new_colors[np.arange(len(reshaped_new_colors)), max_indices]

        reshaped_new_colors_back  = np.array(new_colors_back).reshape(-1, self.pixel_size)
        new_angles_back = np.arctan(np.abs(np.gradient(new_colors_back)))
        reshaped_new_angls_back  = np.array(new_angles).reshape(-1, self.pixel_size)

        averages_angls_back = np.max(reshaped_new_angls_back, axis=1)
        max_indices_back = np.argmax(reshaped_new_angls_back, axis=1)
        averages_colors_back = reshaped_new_colors_back[np.arange(len(reshaped_new_colors_back)), max_indices]

        return averages_colors, averages_colors_back


    def make_flash(self, fig):
        """
        здесь сначала идет расчет максимальной ширины и высоты отверстия

        затем если удовлетворяет - расчитывается засвет
        """
        # print()
        row_width_image, column_width_image, flag_blacks = self.hole_max_width_height(fig.mask_figure_local)

        if row_width_image.max()>=200 or column_width_image.max()>=200 : if_flash = True#if_flash = np.random.choice([True, False], p=[0.7,0.3])
        else: if_flash = False

        if if_flash:
            flash_type = np.random.choice(['vertical', 'horizontal'], p=[0.5,0.5])
            flash_side = np.random.choice(['left', 'center', 'right', 'top', 'bottom'], p=[0.2,0.2,0.2,0.2,0.2])
            percent_flash = random.randint(10, 150)
            percent_law = np.random.choice(['constant', 'linear', 'bezier', 'parabola'], p=[0.25, 0.25, 0.25, 0.25])

            
            if percent_law == 'bezier': growth_rate = random.uniform(1.1, 3.)
            elif percent_law == 'linear': growth_rate = random.uniform(-2., 2.)

            else: growth_rate = random.uniform(-3., 3.)
            # print(flash_type, flash_side, percent_flash, percent_law, growth_rate)
            if column_width_image.max()>=200 and row_width_image.max()<200: flash_type = 'horizontal'
            elif row_width_image.max()>=200 and column_width_image.max()<200: flash_type = 'vertical'

            if flash_side in ['top', 'bottom']: flash_type = 'vertical'
            elif flash_side in ['left', 'right']: flash_type = 'horizontal'

            if flash_type == 'vertical':
                pred_dist = 0
                col_start = np.unique(np.argwhere(fig.mask_figure_local == 128)[:,1])[0]
                col_end = np.unique(np.argwhere(fig.mask_figure_local == 128)[:,1])[-1]

                row_start = np.unique(np.argwhere(fig.mask_figure_local == 128)[:,0])[0]
                row_end = np.unique(np.argwhere(fig.mask_figure_local == 128)[:,0])[-1]

                diff_rows = row_end - row_start

                length_cols = col_end - col_start + 1
                if percent_law == 'constant':
                    percent_list = [percent_flash] * length_cols
                elif percent_law == 'exponential':
                    percent_list = self.exponential_percent_rule(length_cols, percent_flash, growth_rate)
                elif percent_law == 'linear':
                    percent_list = self.linear_percent_rule(length_cols, percent_flash, growth_rate)
                elif percent_law == 'bezier':
                    percent_list = self.bezier_percent_rule(length_cols, percent_flash, growth_rate)
                elif percent_law == 'parabola':
                    percent_list = self.bezier_parabola_percent_rule(length_cols, percent_flash, growth_rate)
                    if len(percent_list) != length_cols:
                        percent_list = percent_list + list([percent_list[-1]])

                for i, col in enumerate(np.arange(col_start, col_end +1)):
                    ind_start = np.unique(np.argwhere(fig.mask_figure_local[:,col] == 128)[:,0])[0]
                    ind_end = np.unique(np.argwhere(fig.mask_figure_local[:,col] == 128)[:,0])[-1]
                    blacks = np.where(fig.mask_figure_local[ind_start:ind_end+1,col] == 0)
                    if flash_side == 'center':
                        if flag_blacks: 
                            dist_ = (row_end - row_start + 1)//2
                        else: dist_ = len(fig.mask_figure_local[ind_start:ind_end+1,col])//2
                    else:
                        if flag_blacks: dist_ = row_end - row_start + 1
                        else: dist_ = len(fig.mask_figure_local[ind_start:ind_end+1,col])

                    new_colors, _ = self.compute_new_colors(dist_, percent=percent_list[i])
                    # print(new_colors)
                    whites_and_bords = np.where(fig.mask_figure_local[ind_start:ind_end+1,col] != 0)
                    if flash_side == 'bottom':
                        fig.color_map_local[ind_start:ind_end+1,col][whites_and_bords] = new_colors[whites_and_bords]
                    elif flash_side == 'top': 
                        fig.color_map_local[ind_start:ind_end+1,col][whites_and_bords] = new_colors[::-1][whites_and_bords]
                    elif flash_side == 'center':
                        new_colors_ = list(new_colors) + list(new_colors[::-1]) + list([new_colors[0]])
                        # new_colors_back_ = list(new_colors_back) + list(new_colors_back[::-1]) + list([new_colors_back[0]])
                        fig.color_map_local[ind_start:ind_end+1,col][whites_and_bords] = np.array(new_colors_)[whites_and_bords]
                        if len(blacks[0]) == 0 and (ind_end - ind_start + 1)//2 < dist_:
                            fig.color_map_local[ind_start:ind_end+1, col][whites_and_bords] = np.array(new_colors_)[whites_and_bords][::-1]
                        else: fig.color_map_local[ind_start:ind_end+1, col][whites_and_bords] = np.array(new_colors_)[whites_and_bords]

            elif flash_type == 'horizontal':
                row_start = np.unique(np.argwhere(fig.mask_figure_local == 128)[:,0])[0]
                row_end = np.unique(np.argwhere(fig.mask_figure_local == 128)[:,0])[-1]

                col_start = np.unique(np.argwhere(fig.mask_figure_local == 128)[:,1])[0]
                col_end = np.unique(np.argwhere(fig.mask_figure_local == 128)[:,1])[-1]

                length_rows = row_end - row_start + 1
                if percent_law == 'constant':
                    percent_list = [percent_flash] * length_rows
                elif percent_law == 'exponential':
                    percent_list = self.exponential_percent_rule(length_rows, percent_flash, growth_rate)
                elif percent_law == 'linear':
                    percent_list = self.linear_percent_rule(length_rows, percent_flash, growth_rate)
                elif percent_law == 'bezier':
                    percent_list = self.bezier_percent_rule(length_rows, percent_flash, growth_rate)
                elif percent_law == 'parabola':
                    percent_list = self.bezier_parabola_percent_rule(length_rows, percent_flash, growth_rate)
                    if len(percent_list) != length_rows:
                        percent_list = percent_list + list([percent_list[-1]])

                for i, row in enumerate(np.arange(row_start, row_end+1)):
                    ind_start = np.unique(np.argwhere(fig.mask_figure_local[row,:] == 128)[:,0])[0]
                    ind_end = np.unique(np.argwhere(fig.mask_figure_local[row,:] == 128)[:,0])[-1]
                    blacks = np.where(fig.mask_figure_local[row, ind_start:ind_end+1] == 0)

                    if flash_side == 'center': 
                        if flag_blacks: dist_ = (col_end - col_start + 1)//2 #dist_ = len(solv.mask_objects[0].signal[row,ind_start:ind_end+1])//2
                        else: dist_ = len(fig.mask_figure_local[row,ind_start:ind_end+1])//2
                    else: 
                        if flag_blacks: dist_ = col_end - col_start + 1 #dist_ = len(solv.mask_objects[0].signal[row,ind_start:ind_end+1])
                        else: dist_ = len(fig.mask_figure_local[row,ind_start:ind_end+1])
                    new_colors, _ = self.compute_new_colors(dist_, percent=percent_list[i])
                    # print(new_colors)

                    whites_and_bords = np.where(fig.mask_figure_local[row,ind_start:ind_end+1] != 0)

                    if flash_side == 'right':
                        fig.color_map_local[row,ind_start:ind_end+1][whites_and_bords] = new_colors[whites_and_bords]

                    elif flash_side == 'left': 
                        fig.color_map_local[row,ind_start:ind_end+1][whites_and_bords] = new_colors[::-1][whites_and_bords]

                    elif flash_side == 'center':
                        new_colors_ = list(new_colors) + list(new_colors[::-1]) + list([new_colors[0]])
                        if len(blacks[0]) == 0 and (ind_end - ind_start + 1)//2 < dist_: 
                            print('here')
                            fig.color_map_local[row,ind_start:ind_end+1][whites_and_bords] = np.array(new_colors_)[whites_and_bords][::-1]
                        else: fig.color_map_local[row,ind_start:ind_end+1][whites_and_bords] = np.array(new_colors_)[whites_and_bords]

            del new_colors, whites_and_bords, blacks, ind_start, ind_end, dist_
            gc.collect()
            # print('AFTER: ', np.unique(fig.color_map_local))
        return fig.color_map_local

    def hole_max_width_height(self, mask):
        flag_blacks = False

        row_width = np.zeros_like(mask, dtype=np.float32)
        row_start = np.argwhere(mask == 255)[0][0]
        rows_count = len(np.unique(np.argwhere(mask == 255)[:, 0]))-1
        row_end = np.unique(np.argwhere(mask == 255)[:, 0])[rows_count]
        col_count = len(np.unique(np.argwhere(mask == 255)[:, 1]))-1

        col_end = np.unique(np.argwhere(mask== 255)[:, 1])[col_count]
        col_start = np.unique(np.argwhere(mask == 255)[:, 1])[0]

        for row in range(row_start, row_end):
            ind_start = np.unique(np.argwhere(mask[row,:] == 128)[:,0])[0]
            ind_end = np.unique(np.argwhere(mask[row,:] == 128)[:,0])[-1]
            
            white_pixels = np.argwhere(mask[row,:] == 255)
            blacks = np.argwhere(mask[row,ind_start:ind_end] == 0)
            if len(blacks) != 0 and not flag_blacks:
                flag_blacks = True 
            for pixel in white_pixels:
                if mask[row,pixel]==255 and mask[row,pixel-1] == 128:
                    start = pixel
                elif mask[row,pixel]==255 and mask[row,pixel+1] == 128:
                    end = pixel
                    row_width[row, start[0]:end[0]+1] = len(row_width[row, start[0]:end[0]]) 

        column_width = np.zeros_like(mask, dtype=np.float32)

        for col in range(col_start, col_end):
            ind_start = np.unique(np.argwhere(mask[:,col] == 128)[:,0])[0]
            ind_end = np.unique(np.argwhere(mask[:,col] == 128)[:,0])[-1]
            white_pixels = np.argwhere(mask[:,col] == 255)
            blacks = np.argwhere(mask[ind_start:ind_end, col] == 0)
            if len(blacks) != 0 and not flag_blacks:
                flag_blacks = True
            for pixel in white_pixels:
                if mask[pixel, col] == 255 and mask[pixel-1, col] == 128:
                    start = pixel
                elif mask[pixel, col] == 255 and mask[pixel+1, col] == 128:
                    end = pixel
                    column_width[start[0]:end[0]+1, col] = len(column_width[start[0]:end[0], col])
        return row_width, column_width, flag_blacks
    