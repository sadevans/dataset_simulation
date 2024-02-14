import numpy as np
import cv2


class Figure():
    def __init__(self, position, hole_contour, border_contour, image):
        self.position = position
        self.hole_contour = hole_contour.copy()
        self.border_contour = border_contour.copy()

        self.color_map_local = np.zeros_like(image, dtype=np.float32)
        self.angles_map_local = np.zeros_like(image, dtype=np.float32)
        self.width_map_local = np.zeros_like(image, dtype=np.float32)

        self.mask_figure_local = np.zeros_like(image, dtype=np.float32)

        self.dp2 = (1, 0)
        self.dp3 = (0, 1)


    def draw(self, image, args, position, border_width):
        if 'rectangle' in args['shape']: 
            return self.draw_rect(image, position, args['height'], args['width'], args['corner_radius'], border_width)
        elif 'circle' in args['shape']:
            return self.draw_circle(image, position,  args['height'], border_width)


    def draw_rect(self, image, position, height, width, corner_radius, border_width):
        temp = np.zeros_like(image, dtype=np.float32)
        temp = cv2.rectangle(temp, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
                (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, 20) 
        temp = cv2.rectangle(temp, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
                    (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, -1)
    
        cont = self.init_contour(temp)
        cv2.drawContours(image, cont, 0, 128, border_width*2)
        cv2.drawContours(temp, cont, 0, 128, border_width*2)
        bord_cont = self.init_contour(temp)
        image = cv2.rectangle(image, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
                (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, 20) 
        image = cv2.rectangle(image, (position[0] - (height//2-corner_radius//2), position[1]-(height//2-corner_radius//2)), 
                    (position[0] + (height//2-corner_radius//2), position[1]+ (height//2-corner_radius//2)), 255, -1)
        

        self.hole_contour = cont[0].reshape(-1, 2).copy()
        self.border_contour = bord_cont[0].reshape(-1, 2).copy()


        self.make_local_mask()
        return image
    

    def draw_circle(self, image, position, height, border_width):
        temp = np.zeros_like(image, dtype=np.float32)

        cv2.circle(temp, position, height//2, 255, 2)
        cv2.circle(temp, position, height//2, 255, -1)
        cont = self.init_contour(temp)
        cv2.drawContours(temp, cont, 0, 128, border_width*2)
        bord_cont = self.init_contour(temp)
        cv2.drawContours(image, cont, 0, 128, border_width*2)
        cv2.circle(image, position, height//2, 255, 2)
        cv2.circle(image, position, height//2, 255, -1)

        self.hole_contour = cont[0].reshape(-1, 2).copy()
        self.border_contour = bord_cont[0].reshape(-1, 2).copy()

        self.make_local_mask()
        return image


    def make_local_mask(self):
        cv2.drawContours(self.mask_figure_local, [self.border_contour], 0, 128, -1)
        cv2.drawContours(self.mask_figure_local, [self.hole_contour], 0, 255, -1)


    def init_contour(self, image):
        cont, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont


    def compute_local_maps(self, solver):
        self.width_map_local, self.angles_map_local, self.color_map_local = solver.transform_bezier(self.mask_figure_local, \
                                                                                [self.border_contour], [self.hole_contour])


    def make_flash(self, solver):
        # print('BEFORE: ', np.unique(self.color_map_local))
        self.color_map_local = solver.make_flash(self).copy()
        # print(np.unique(self.color_map_local))




    def detect_cont(self, img):
        cont, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cont
